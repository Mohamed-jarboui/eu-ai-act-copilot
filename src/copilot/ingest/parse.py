"""Parse the Cellar XHTML of a regulation into Units.

The OJ XHTML is regular: recitals are `div#rct_N`, articles `div#art_N` (title in `art_N.tit_1`),
numbered paragraphs `div#PPP.NNN`, points are nested two-cell tables (label | text), annexes are
`div#anx_ROMAN`. Footnote markers (`a#ntc...`) and footnote bodies (`p.oj-note`) are dropped.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from copilot.ingest.models import Unit
from copilot.ingest.refs import extract_cross_refs

NS = "{http://www.w3.org/1999/xhtml}"

# (regulation -> article that holds the definitions)
DEFINITIONS_ARTICLE = {"aiact": "3", "gdpr": "4"}

_TITLE_CLASSES = {"oj-ti-art", "oj-sti-art"}
_PARAGRAPH_ID = re.compile(r"\d{3}\.(\d{3})$")
_LABEL = re.compile(r"^\(?\w{1,6}\)$|^\d{1,3}\.$")  # "(a)", "a)" (FR/DE), "(12)", "3."
_BULLET = re.compile(r"^[–—\-•]$")
_QUOTED_TERM = (
    re.compile(r"‘(.+?)’(?=[\s,:;]|$)"),  # EN, FR, DE single quotes
    re.compile(r"«\s*(.+?)\s*»"),  # FR guillemets
    re.compile(r"„(.+?)“"),  # DE low-high quotes
)


def _tag(el: ET.Element) -> str:
    return el.tag.removeprefix(NS) if isinstance(el.tag, str) else ""


def _classes(el: ET.Element) -> list[str]:
    return (el.get("class") or "").split()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _inline(el: ET.Element) -> str:
    """Flattened text of an element, minus footnote markers."""
    parts: list[str] = []

    def walk(e: ET.Element) -> None:
        if e.text:
            parts.append(e.text)
        for child in e:
            if _tag(child) == "a" and (child.get("id") or "").startswith("ntc"):
                pass  # footnote marker such as "(1)"
            elif _tag(child) == "br":
                parts.append(" ")
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return _clean("".join(parts))


def _is_block(el: ET.Element) -> bool:
    return any(_tag(c) in ("p", "table", "div") for c in el)


def _lines(el: ET.Element) -> list[str]:
    """Text of an element's children as lines; table rows render as '(a) text'."""
    out: list[str] = []
    for child in el:
        tag = _tag(child)
        classes = _classes(child)
        if tag == "p":
            if "oj-note" in classes or _TITLE_CLASSES & set(classes):
                continue
            if text := _inline(child):
                out.append(text)
        elif tag == "table":
            out.extend(_table_lines(child))
        elif tag == "div":
            if "eli-title" in classes:
                continue
            if _is_block(child):
                out.extend(_lines(child))
            elif text := _inline(child):
                out.append(text)
    return out


def _rows(table: ET.Element) -> list[ET.Element]:
    body = table.find(f"{NS}tbody")
    return [r for r in (body if body is not None else table) if _tag(r) == "tr"]


def _cells(row: ET.Element) -> list[ET.Element]:
    return [c for c in row if _tag(c) == "td"]


def _row_label(row: ET.Element) -> str | None:
    """The label of a (label | text) row, e.g. "(a)", or None if it is not such a row."""
    cells = _cells(row)
    if len(cells) != 2:
        return None
    label = _inline(cells[0])
    return label if _LABEL.match(label) or _BULLET.match(label) else None


def _table_lines(table: ET.Element) -> list[str]:
    out: list[str] = []
    for row in _rows(table):
        cells = _cells(row)
        label = _row_label(row)
        if label is not None:
            body = _lines(cells[1]) or ([_inline(cells[1])] if _inline(cells[1]) else [])
            if body:
                body[0] = f"{label} {body[0]}"
                out.extend(body)
        else:
            for cell in cells:
                out.extend(_lines(cell))
    return out


def _label_key(label: str) -> str:
    return label.strip("().").lower()


@dataclass
class _Ctx:
    regulation: str
    language: str
    root: ET.Element
    parents: dict[ET.Element, ET.Element]
    units: list[Unit] = field(default_factory=list)
    _chapters: dict[ET.Element, str | None] = field(default_factory=dict)
    list_state: dict[str, list] = field(default_factory=dict)  # parent id -> [list no, seen keys]

    def add(self, **kwargs) -> Unit:
        text = kwargs["text"]
        unit = Unit(
            regulation=self.regulation,
            language=self.language,
            refs=tuple(extract_cross_refs(text, self.language)),
            **kwargs,
        )
        self.units.append(unit)
        return unit

    def chapter_of(self, el: ET.Element) -> str | None:
        node: ET.Element | None = el
        while node is not None:
            if re.fullmatch(r"cpt_[IVXLC]+", node.get("id") or ""):
                if node not in self._chapters:
                    self._chapters[node] = self._chapter_label(node)
                return self._chapters[node]
            node = self.parents.get(node)
        return None

    @staticmethod
    def _chapter_label(chapter: ET.Element) -> str:
        label = next((_inline(p) for p in chapter if "oj-ti-section-1" in _classes(p)), "")
        title_div = next((c for c in chapter if (c.get("id") or "").endswith(".tit_1")), None)
        title = _inline(title_div) if title_div is not None else ""
        return f"{label} - {title}" if title else label


def _title_of(el: ET.Element, id_prefix: str) -> str | None:
    div = next((c for c in el if c.get("id") == f"{id_prefix}.tit_1"), None)
    return _inline(div).rstrip("` ") or None if div is not None else None


def _points(ctx: _Ctx, container: ET.Element, parent: Unit, article: str | None, annex: str | None):
    """Emit one `point` unit per labelled table row in container, recursing into sub-points."""
    for child in container:
        tag = _tag(child)
        if tag == "div" and not child.get("id") and "eli-title" not in _classes(child):
            _points(ctx, child, parent, article, annex)
        elif tag == "table":
            for row in _rows(child):
                label = _row_label(row)
                if label is None or _BULLET.match(label):
                    continue
                cell = _cells(row)[1]
                body = _lines(cell) or ([_inline(cell)] if _inline(cell) else [])
                key = _label_key(label)
                # A parent can hold several lists reusing the same labels (e.g. AI Act Art. 43(1)):
                # the first list keeps plain ids, later ones get an "lN:" prefix.
                state = ctx.list_state.setdefault(parent.id, [1, set()])
                if key in state[1]:
                    state[0] += 1
                    state[1] = set()
                state[1].add(key)
                prefix = f"l{state[0]}:" if state[0] > 1 else ""
                unit = ctx.add(
                    id=f"{parent.id}:{prefix}{key}",
                    type="point",
                    article=article,
                    annex=annex,
                    number=key,
                    title=None,
                    chapter=parent.chapter,
                    text="\n".join(body),
                    parent_id=parent.id,
                )
                _points(ctx, cell, unit, article, annex)


def _term_of(text: str) -> str:
    for pattern in _QUOTED_TERM:
        if match := pattern.search(text):
            return match.group(1).strip()
    return text[:60]


def _parse_recitals(ctx: _Ctx) -> None:
    for el in ctx.root.iter():
        match = re.fullmatch(r"rct_(\d+)", el.get("id") or "")
        if _tag(el) != "div" or not match:
            continue
        text = re.sub(r"^\(\d+\)\s*", "", "\n".join(_lines(el)))
        ctx.add(
            id=f"{ctx.regulation}:rct:{match.group(1)}",
            type="recital",
            article=None,
            annex=None,
            number=match.group(1),
            title=None,
            chapter=None,
            text=text,
            parent_id=None,
        )


def _parse_articles(ctx: _Ctx) -> None:
    for el in ctx.root.iter():
        match = re.fullmatch(r"art_(\d+[a-z]?)", el.get("id") or "")
        if _tag(el) != "div" or not match:
            continue
        number = match.group(1)
        chapter = ctx.chapter_of(el)
        article = ctx.add(
            id=f"{ctx.regulation}:art:{number}",
            type="article",
            article=number,
            annex=None,
            number=number,
            title=_title_of(el, el.get("id")),
            chapter=chapter,
            text="\n".join(_lines(el)),
            parent_id=None,
        )
        if DEFINITIONS_ARTICLE.get(ctx.regulation) == number:
            _parse_definitions(ctx, el, article)
            continue
        numbered = [c for c in el if _tag(c) == "div" and _PARAGRAPH_ID.match(c.get("id") or "")]
        if not numbered:  # unnumbered body: points hang directly off the article
            _points(ctx, el, article, number, None)
            continue
        for div in numbered:
            pnum = str(int(_PARAGRAPH_ID.match(div.get("id")).group(1)))
            text = re.sub(r"^\(?\d+[.)]\s*", "", "\n".join(_lines(div)))  # "1." or "(1)" (DE)
            paragraph = ctx.add(
                id=f"{article.id}:{pnum}",
                type="paragraph",
                article=number,
                annex=None,
                number=pnum,
                title=article.title,
                chapter=chapter,
                text=text,
                parent_id=article.id,
            )
            _points(ctx, div, paragraph, number, None)


def _parse_definitions(ctx: _Ctx, art_el: ET.Element, article: Unit) -> None:
    for table in (c for c in art_el if _tag(c) == "table"):
        for row in _rows(table):
            label = _row_label(row)
            if label is None or _BULLET.match(label):
                continue
            body = "\n".join(_lines(_cells(row)[1]))
            key = _label_key(label)
            ctx.add(
                id=f"{ctx.regulation}:def:{key}",
                type="definition",
                article=article.article,
                annex=None,
                number=key,
                title=_term_of(body),
                chapter=article.chapter,
                text=body,
                parent_id=article.id,
            )


@dataclass
class _Segment:
    kind: str  # "container" | "item"
    key: str
    container: str | None
    title: str | None
    lines: list[str] = field(default_factory=list)


def _annex_segments(children: list[ET.Element]) -> list[_Segment]:
    """Split an annex body into sections: 'Section A' containers and numbered/lettered items."""

    def classify(el: ET.Element) -> tuple[str, str | None]:
        tag, classes = _tag(el), _classes(el)
        if tag == "p" and "oj-ti-grseq-1" in classes:
            text = _inline(el)
            if m := re.match(r"^(\d+)\.\s", text):
                return "item", m.group(1)
            if m := re.match(r"^(?!\d)[^\W\d_]+\s+([A-Z]|\d+)\b", text):
                return "container", m.group(1)
        elif tag == "table":
            rows = _rows(el)
            label = _row_label(rows[0]) if rows else None
            if label and re.fullmatch(r"\d{1,3}[.)]", label):
                return "item", label.rstrip(".)")
            if label and re.fullmatch(r"\(?[a-z]{1,4}\)", label):
                return "letter", _label_key(label)
        elif tag == "div" and "oj-enumeration-spacing" in classes:
            if m := re.match(r"^(\d+)\.\s", _inline(el)):
                return "item", m.group(1)
        return "content", None

    kinds = [classify(c) for c in children]
    use_letters = not any(k == "item" for k, _ in kinds)
    segments: list[_Segment] = []
    container: _Segment | None = None
    current: _Segment | None = None
    for child, (kind, key) in zip(children, kinds, strict=True):
        if kind == "letter" and not use_letters:
            kind = "content"
        elif kind == "letter":
            kind = "item"
        lines = _child_lines(child)
        if kind == "container":
            container = current = _Segment("container", key, None, None, [_inline(child)])
            segments.append(current)
        elif kind == "item":
            heading = _tag(child) == "p" or _tag(child) == "div"
            title = re.sub(r"^\d+\.\s*", "", _inline(child)) if heading else None
            current = _Segment(
                "item", key, container.key if container else None, title, lines or [_inline(child)]
            )
            segments.append(current)
            if container:
                container.lines.extend(current.lines)
        elif current is not None:
            current.lines.extend(lines)
            if container and current is not container:
                container.lines.extend(lines)
    return segments


def _child_lines(child: ET.Element) -> list[str]:
    """Lines of a single child element (reusing _lines, which iterates over a parent's children)."""
    holder = ET.Element("holder")
    holder.append(child)
    return _lines(holder)


def _parse_annexes(ctx: _Ctx) -> None:
    for el in ctx.root.iter():
        match = re.fullmatch(r"anx_([IVXLC]+)", el.get("id") or "")
        if _tag(el) != "div" or not match:
            continue
        numeral = match.group(1)
        heads = [c for c in el if "oj-doc-ti" in _classes(c)]
        body = [c for c in el if c not in heads[:2]]
        annex = ctx.add(
            id=f"{ctx.regulation}:anx:{numeral}",
            type="annex",
            article=None,
            annex=numeral,
            number=numeral,
            title=_inline(heads[1]) if len(heads) > 1 else None,
            chapter=None,
            text="\n".join(line for c in body for line in _child_lines(c)),
            parent_id=None,
        )
        for seg in _annex_segments(body):
            path = f"{seg.container}:{seg.key}" if seg.container else seg.key
            ctx.add(
                id=f"{annex.id}:{path}",
                type="annex_section",
                article=None,
                annex=numeral,
                number=seg.key,
                title=seg.title,
                chapter=None,
                text="\n".join(seg.lines),
                parent_id=annex.id,
            )


def parse_file(path: Path, regulation: str, language: str) -> list[Unit]:
    root = ET.parse(path).getroot()
    parents = {child: parent for parent in root.iter() for child in parent}
    ctx = _Ctx(regulation, language, root, parents)
    _parse_recitals(ctx)
    _parse_articles(ctx)
    _parse_annexes(ctx)
    return ctx.units
