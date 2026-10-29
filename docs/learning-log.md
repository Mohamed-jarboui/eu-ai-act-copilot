# Learning log

Everything left "for later" in learning mode: the `TODO(me)` pieces and the understanding questions, each with an answer and the reasoning. Try each item yourself first, then compare.

## Open TODO(me) items

| # | Stage | File | What to implement | Tests that tell you when you're done |
|---|---|---|---|---|
| 1 | 0 | `src/copilot/llm/router.py` | `should_fallback(error)`: which errors move on to the next provider | `tests/test_llm_router.py` (add an `AuthError` case) |
| 2 | 1 | `src/copilot/ingest/refs.py` | `extract_cross_refs(text, lang)`: find "Article 9" / "Annex III" in EN, FR, DE | `tests/test_refs.py` (8 xfail cases: remove each marker as it passes) |

After finishing #2, run `make ingest` (the corpus stores `refs`, and `test_committed_corpus_is_up_to_date` fails if the committed corpus is stale), then update the "Cross-references" bullet in `docs/data-card.md`.

---

## Stage 0: provider layer

### TODO #1: `should_fallback`

**The question to answer:** after a provider is finished with (retries used up, or the error was not retryable), would the *next* provider plausibly succeed with the same request?

| Error | Retry same provider? (`is_retryable`) | Fall back? (`should_fallback`) | Why |
|---|---|---|---|
| `RateLimitError` (429) | yes, with backoff | yes | Gemini's limit says nothing about Groq's |
| `ServerError` (5xx, timeout, connection) | yes, with backoff | yes | The outage is on one provider's side |
| `AuthError` (401/403) | no, a bad key stays bad | **your call**, see below | Groq has its own key |
| `BadRequestError` (other 4xx) | no | no | The request itself is wrong; the next provider would reject it too |

**Reference answer**

```python
def should_fallback(error: ProviderError) -> bool:
    if isinstance(error, BadRequestError):
        return False
    return isinstance(error, RateLimitError | ServerError | AuthError)
```

**Explanation**

- `AuthError` is the interesting one. Falling back keeps the product answering when a Gemini key is revoked or mistyped, which is what a user wants. The cost is *silent degradation*: you could run on Groq for weeks without noticing Gemini is broken. The compromise is to fall back **and** make the failure visible. Every failed attempt is already in `llm.usage` with its `error` string, so a later monitoring step can alert on repeated `AuthError`s. The stricter alternative (return `False`, fail loudly) is also defensible, especially in development.
- `BadRequestError` has one subtlety: a request one model rejects (for example a prompt larger than its context window) could be accepted by a model with a bigger one. The simple rule "bad request never falls back" is a deliberate simplification. If it ever bites, split that case into its own error type.
- `AllProvidersFailedError` is raised by the router itself and never reaches this function. `UnsupportedOperationError` is handled before it (the provider is skipped without counting as a failure).

**Tests to add** (the placeholder does not cover `AuthError`):

```python
def test_auth_error_falls_back_to_next_provider():
    primary = FakeProvider("gemini", [AuthError("401")])
    backup = FakeProvider("groq", ["from groq"])
    llm, sleeps = make_llm(primary, backup)
    assert llm.generate("hi").text == "from groq"
    assert primary.calls == 1  # not retried
    assert sleeps == []
```

If you chose the strict alternative, assert `pytest.raises(AuthError)` and `backup.calls == 0` instead.

### Questions and answers

**1. Why are `is_retryable` and `should_fallback` separate decisions? Give an error where they disagree.**
`is_retryable` asks "might the same provider succeed on the next attempt?" (transient problems). `should_fallback` asks "would a different provider succeed?" (provider-specific problems). They disagree on `AuthError`: retrying a bad key is pointless, but another provider with its own key can work. They also disagree in time on `RateLimitError`: it is retryable first, and only after the retries are used up does it trigger fallback. `BadRequestError` is no on both.

**2. Why `max_retries=0` on the Groq client?**
The Groq SDK retries on its own by default. Hidden retries would make one router attempt hide several HTTP requests (3 router attempts could become 9), so the failed requests would never appear in the usage records, the latency would include invisible waiting, and the backoff timing would no longer be the router's, which makes it untestable. With `0`, all retrying is in the router, where it is visible, logged and tested. Caveat: the Gemini client was not configured for retries; I believe `google-genai` does not retry by default but did not verify it, so check that before trusting the usage log to be exact.

**3. Why inject `sleep` and `clock` instead of patching `time`? How does the latency assertion depend on it?**
Injection makes the tests instant (no real sleeping), deterministic, and able to assert exact backoff values (`sleeps == [1.0, 2.0]`, which is `base * 2**attempt`), without monkeypatching global state that could leak into other tests. The fake clock advances 1 second on every reading. The router reads it exactly twice per attempt (start and end), so `latency_s` is exactly `1.0`. With the real `perf_counter` it would be microseconds and the assertion flaky. The catch: the test depends on the router reading the clock exactly twice per attempt; a third reading would break it.

---

## Stage 1: ingestion

### TODO #2: `extract_cross_refs`

**Target output** (language-independent, regulation-agnostic): `"art:9"`, `"art:6:2"` (paragraph 2 of Article 6), `"anx:III"`, in order of first appearance, no duplicates.

**Patterns to cover**

| | Article | Paragraph | List | Annex |
|---|---|---|---|---|
| EN | `Article 9` | `Article 6(2)` | `Articles 9 and 10` | `Annex III`, `Annexes I and III` |
| FR | `article 9` | `article 6, paragraphe 2` | `articles 9 et 10` | `annexe III` |
| DE | `Artikel 9` | `Artikel 6 Absatz 2` | `die Artikel 9 und 10` | `Anhang III`, `Anhänge I und III` |

**Design decisions to make** (the reference answer makes these choices; you may choose differently, but write them down in the data card):

1. **Case:** keywords are matched case-insensitively (`Article`/`article`/`Artikel`), but Roman numerals stay case-sensitive so `Annex it` is not read as an annex.
2. **Lists:** capture the whole run of numbers after the keyword, then split it. Without this, "Articles 9 and 10" yields only `art:9`.
3. **Ranges:** "Articles 8 to 15" is expanded to 8, 9, ..., 15, because the text does refer to each of them. (Alternative: keep only the endpoints.)
4. **Other acts:** "Article 6 of Regulation (EU) 2016/679" is **skipped**. Recording it as `art:6` would point at Article 6 of the AI Act, a different article. The same goes for `Article 16 TFEU` and `Article 2, point (4) of Directive ...`. "of this Regulation" is kept.
5. **Regex trap:** do not put "not followed by of Regulation" in a lookahead inside the same regex. A greedy match that fails the lookahead backtracks to a shorter match ("Articles 9 and 10 of Directive..." would still yield `art:9`). Match first, then inspect the text after the match.

**Reference answer** (about 80 lines; it passes your 11 test cases plus 12 extras: ranges, other-act exclusion, `43a`, French "paragraphe", German "Anhänge").

<details>
<summary>Show the reference implementation (try it yourself first)</summary>

```python
from __future__ import annotations

import re

# Per-language words: article / annex keyword, "paragraph" word, "and/or" words, "to" (range).
_KW = {
    "en": dict(art=r"Articles?", anx=r"Annex(?:es)?", para=r"paragraph", conj=r"and|or", to=r"to"),
    "fr": dict(art=r"articles?", anx=r"annexes?", para=r"paragraphe", conj=r"et|ou", to=r"à"),
    "de": dict(
        art=r"Artikel[n]?", anx=r"Anh(?:ang|änge[n]?)", para=r"Absatz", conj=r"und|oder", to=r"bis"
    ),
}

# "... of Regulation (EU) 2016/679" and similar: the target is NOT in this text.
_OTHER_ACT = {
    "en": (
        r"of\s+(?:the\s+)?(?:Council\s+|Commission\s+)?"
        r"(?:(?:Delegated\s+|Implementing\s+)?(?:Regulation|Directive|Decision)"
        r"|Treaty|TFEU|that\s+(?:Regulation|Directive))\b"
    ),
    "fr": r"(?:du|de\s+la|des|de\s+l’)\s*(?:règlement|directive|décision|traité|TFUE)\b",
    "de": r"(?:der|des)\s+(?:Verordnung|Richtlinie|Beschlusses|Entscheidung|Vertrags?|AEUV)\b",
}
_TREATY_ABBREV = r"(?:TFEU|TEU|TFUE|TUE|AEUV|EUV)\b"
# The clause that follows a reference ends at a sentence break or at the next Article/Annex.
_CLAUSE_END = r"[.;]\s|\b(?i:articles?|artikel\w*|annex\w*|anh\w+)\b"


def _build(lang: str, kind: str) -> re.Pattern:
    k = _KW[lang]
    sep = rf"(?:\s*,\s*(?:(?i:{k['conj']})\s+)?|\s+(?i:{k['conj']})\s+|\s+(?i:{k['to']})\s+)"
    if kind == "art":
        item = rf"\d+[a-z]?(?:\(\d+\)|,?\s*(?i:{k['para']})\s+\d+)?"
    else:
        item = r"[IVXLC]+\b"
    return re.compile(rf"(?i:{k[kind]})\s+(?P<span>{item}(?:{sep}{item})*)")


_PATTERNS = {(lang, kind): _build(lang, kind) for lang in _KW for kind in ("art", "anx")}
_TOKEN = {
    lang: re.compile(rf"(?P<n>\d+[a-z]?)(?:\((?P<p1>\d+)\)|,?\s*(?i:{k['para']})\s+(?P<p2>\d+))?")
    for lang, k in _KW.items()
}


def _is_other_act(tail: str, lang: str) -> bool:
    """Does the text right after a reference say it belongs to another act (Directive, TFEU...)?"""
    clause = re.split(_CLAUSE_END, tail, maxsplit=1)[0][:90]
    if re.match(rf"\s*{_TREATY_ABBREV}", clause):
        return True
    return bool(re.search(_OTHER_ACT[lang], clause, re.IGNORECASE))


def _article_refs(span: str, lang: str) -> list[str]:
    refs, prev = [], None
    tokens = list(_TOKEN[lang].finditer(span))
    for i, tok in enumerate(tokens):
        num, para = tok.group("n"), tok.group("p1") or tok.group("p2")
        ref = f"art:{num}:{para}" if para else f"art:{num}"
        between = span[tokens[i - 1].end() : tok.start()] if i else ""
        is_range = re.fullmatch(rf"\s*(?i:{_KW[lang]['to']})\s+", between)
        if i and is_range and prev and num.isdigit():  # "Articles 8 to 15" -> 8, 9, ..., 15
            refs.extend(f"art:{n}" for n in range(int(prev) + 1, int(num)))
        refs.append(ref)
        prev = num if num.isdigit() and not para else None
    return refs


def extract_cross_refs(text: str, lang: str) -> list[str]:
    refs: list[str] = []
    for kind in ("art", "anx"):
        for m in _PATTERNS[(lang, kind)].finditer(text):
            if _is_other_act(text[m.end() :], lang):
                continue  # refers to another act
            found = (
                _article_refs(m.group("span"), lang)
                if kind == "art"
                else [f"anx:{r}" for r in re.findall(r"[IVXLC]+\b", m.group("span"))]
            )
            refs.extend(r for r in found if r not in refs)
    return refs
```

</details>

**What I measured with this reference implementation** (run once over the corpus built from `data/raw`, in a scratch script; the repository code was not changed):

- 23 of 23 hand-written cases pass.
- Units with at least one reference: EN 773, FR 777, DE 743, each out of 2375 units per language (AI Act and GDPR together).
- Consistency check: for the same unit id, EN and FR produce identical reference sets on 2330 of 2375 ids (45 differ), and EN and DE on 2250 of 2375 (125 differ). I did not investigate the remaining differences. Looking at the sampled ones (Recitals 40, 41, 136) is a good exercise: a sub-paragraph pattern such as "Article 5(1), first subparagraph, point (g)", or a wording difference between languages, is a likely cause. The cross-language diff is a free test oracle, since the same unit should cite the same targets.

**Known gaps, even in the reference answer:** it does not resolve "of that Regulation" to a specific act, it does not record *which* regulation a skipped reference pointed at (a later improvement is a `gdpr:art:6` style target), and "point (g)" and "subparagraph" levels are not captured.

**Checklist when you finish**

- [ ] Remove the xfail marker from each case in `tests/test_refs.py` that now passes.
- [ ] `make ingest`, then commit the regenerated `data/processed/corpus.jsonl`.
- [ ] Replace the "Cross-references" bullet in `docs/data-card.md` with what your function really covers, with counts from a script run.

### Questions and answers

**1. Ids like `aiact:art:6:2` are the same in EN, FR and DE. What does that buy later stages, and where is uniqueness enforced?**
One gold label works for all three languages: the retrieval evaluation set (question to correct article ids) is written once and scored on EN, FR and DE corpora. Citations stay consistent: an answer in French cites `aiact:art:6:2`, which any renderer can show as "Article 6(2)". Cross-lingual retrieval (a French question, an English passage) can be checked by id equality, and the `cross_reference` tool in `docs/design.md` can follow `refs` in any language. The id alone is **not** globally unique: the key is `(regulation, language, id)`. That is enforced by `check_unique()` in `corpus.py` (called by `make ingest`) and by `test_ids_unique_per_regulation_and_language`. Alignment across languages is enforced by `test_ids_aligned_across_languages`.

**2. FR and DE label points `a)` and definitions `1)` / `1.`, where EN uses `(a)`. Why did the first run give 0 points in French, and which test would have caught it?**
The label pattern `_LABEL` required an opening parenthesis, so `_row_label` returned `None` for any French or German row labelled `a)`, `1)` or `1.`. On the first run (counts printed by a scratch script) French had 0 points and 0 definitions, and German had 6 points (EN had 388) but did get its definitions (68 in the AI Act, 26 in GDPR), because German numbers those `1.`, a shape the old pattern did accept. Only its `a)` points were missed. The golden-count test `test_counts_are_stable` (parametrized per language) fails for FR and DE, and `test_ids_aligned_across_languages` fails because the FR and DE id sets are missing the point and definition ids; `test_aiact_article_5_has_its_points` would fail too. The lesson: tests parametrized over all three languages catch bugs that only appear in one.

**3. How will you treat "Article 6 of Regulation (EU) 2016/679" inside an AI Act text? What goes wrong if you record it as `art:6`?**
Skip it (or, better in a later version, record it with its target regulation). Recording it as plain `art:6` creates a wrong edge: it points to Article 6 of the AI Act (the high-risk classification rule), not Article 6 of GDPR (lawfulness of processing). The `cross_reference` tool would then return a misleading neighbour, graph-style retrieval expansion would pull in an irrelevant article, and a generated answer could cite the wrong article, which is exactly the kind of error the project is meant to avoid. This is why the `refs` format is documented as regulation-agnostic and why `docs/data-card.md` lists it as a limitation.

---

## Other things to do later

- **Licence line:** read the current EUR-Lex legal notice (https://eur-lex.europa.eu/content/legal-notice/legal-notice.html) and replace the "to verify" sentence in `docs/data-card.md` with what it really says.
- **Model names:** the default Gemini and Groq model names in `src/copilot/llm/gemini.py` and `groq.py` were not verified against the providers' current lists. Check them before the first live call.
- **Gemini retries:** confirm that the `google-genai` client does not retry on its own (see Stage 0, question 2).
