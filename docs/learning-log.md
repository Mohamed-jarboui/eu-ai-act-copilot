# Learning log

Everything left "for later" in learning mode: the `TODO(me)` pieces and the understanding questions, each with an answer and the reasoning.

## Status of the TODO(me) items: all implemented

On 2026-10-29 the owner asked for the `TODO(me)` items to be implemented so the project could be finished and merged, so they are **no longer open**: the code contains the implementations (written by Claude, not by the owner) and no `TODO(me)` markers remain. The sections below stay as study material: for each item, what the function has to do, the reasoning, and the design decisions worth understanding. The useful way to use them now is to read the code, close the file, and try to explain or re-derive each decision.

| # | Stage | Implemented in | Decision worth understanding |
|---|---|---|---|
| 1 | 0 | `llm/router.py` (`should_fallback`), `llm/base.py` | Fall back on rate limit, server error, **not found** and **auth** errors; never on a bad request. A 404 now has its own error type (`NotFoundError`) because a retired model is a provider problem, not a bad request. Falling back on auth errors keeps the service up but can hide a broken key, so failures stay visible in `usage`. |
| 2 | 1 | `ingest/refs.py` (`extract_cross_refs`) | Lists and ranges are expanded; references to other acts are skipped. The corpus was regenerated with real `refs` in all three languages. |
| 3-5 | 2 | `retrieve/fusion.py`, `retrieve/metrics.py` | RRF uses ranks, not scores; recall counts each gold id once. The BM25 numbers are reproduced exactly (0.694 / 0.797 / 0.652). |
| 6 | 3 | `answer/citations.py` (`parse_citation`) | The inverse of `labels.label_for`; every one of the 1092 corpus labels round-trips. Unknown labels return `None` so the sentence is dropped, not guessed. |
| 7 | 3 | `answer/refusal.py` (`should_refuse`) | **The weakest piece.** English-only cosine threshold 0.65, calibrated on 9 in-scope questions (0.699 to 0.814) and 1 out-of-scope one (0.599). Provisional, because the embedding quota blocked more data. BM25 scores are not used. |

Still needing a human decision: item 7's threshold (add out-of-scope questions, embed them, recalibrate with `python -m copilot.answer.calibrate --embed`; `evals/refusal_v0.jsonl` holds 12 English ones that are not embedded yet).

---

## Stage 0: provider layer

### TODO #1: `should_fallback` (implemented)

*Status: implemented, with `AuthError` and a new `NotFoundError` falling back; tests in `tests/test_llm_router.py`. The text below is the original task description and reasoning.*

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

### TODO #2: `extract_cross_refs` (implemented)

*Status: implemented in `ingest/refs.py` (the version described here); the corpus and the data card are regenerated.*

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

**What I measured with this reference implementation** (first in a scratch script, then reproduced exactly by `make ingest` once the implementation was in the repository):

- 23 of 23 hand-written cases pass.
- Units with at least one reference: EN 773, FR 777, DE 743, each out of 2375 units per language (AI Act and GDPR together).
- Consistency check: for the same unit id, EN and FR produce identical reference sets on 2330 of 2375 ids (45 differ), and EN and DE on 2250 of 2375 (125 differ). I did not investigate the remaining differences. Looking at the sampled ones (Recitals 40, 41, 136) is a good exercise: a sub-paragraph pattern such as "Article 5(1), first subparagraph, point (g)", or a wording difference between languages, is a likely cause. The cross-language diff is a free test oracle, since the same unit should cite the same targets.

**Known gaps, even in the reference answer:** it does not resolve "of that Regulation" to a specific act, it does not record *which* regulation a skipped reference pointed at (a later improvement is a `gdpr:art:6` style target), and "point (g)" and "subparagraph" levels are not captured.

**Checklist when you finish**

- [x] Remove the xfail marker from each case in `tests/test_refs.py` that now passes.
- [x] `make ingest`, then commit the regenerated `data/processed/corpus.jsonl`.
- [x] Replace the "Cross-references" bullet in `docs/data-card.md` with what your function really covers, with counts from a script run.

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

---

## Stage 2: retrieval

### Status: the vector numbers are not in yet

The first live run embedded all English chunks (1084 distinct texts) and then hit Gemini's free-tier **daily** limit (1000 embedding requests per day, "retry in about 16h"). The only results so far are BM25 on all 37 scored questions (`results/retrieval_bm25_all.json`). To finish, rerun the commands in `evals/README.md` ("Finishing the vector results") once per day, or enable billing on the key (your call). Until then there is no honest answer to "which method wins".

### TODO #3, #4, #5: `rrf_fuse`, `recall_at_k`, `reciprocal_rank` (implemented)

| # | File | Formula |
|---|---|---|
| 3 | `src/copilot/retrieve/fusion.py` | `score(d) = sum over rankings r containing d of 1 / (k + rank_r(d))`, rank starting at 1, `k = 60`; sort by score, ties in first-seen order |
| 4 | `src/copilot/retrieve/metrics.py` | `recall@k = |{g in gold : some retrieved r in the top k has is_hit(r, g)}| / |gold|` |
| 5 | same | `RR = 1 / rank of the first hit` (0 if none); `MRR = mean(RR)` |

*Status: implemented in `fusion.py` and `metrics.py`, and the evaluation runner uses them (`IMPL = "mine"`). Rerunning the BM25 evaluation with them reproduced exactly the numbers below, which is the check for #4 and #5; #3 has tests with exact expected scores and is only exercised by the hybrid runs.*

| Method | recall@5 | recall@10 | MRR |
|---|---|---|---|
| bm25 | 0.694 | 0.797 | 0.652 |

(`python evals/run_retrieval.py --methods bm25 --tag bm25_all` with your implementation should print exactly these.) #3 can only be checked against the vector runs, but its tests have exact expected scores.

<details>
<summary>The implementations (now in <code>fusion.py</code> and <code>metrics.py</code>)</summary>

```python
def rrf_fuse(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])  # sorted() is stable: ties keep order


def recall_at_k(ranked, gold, k):
    top = ranked[:k]
    return sum(1 for g in gold if any(is_hit(r, g) for r in top)) / len(gold)


def reciprocal_rank(ranked, gold):
    for rank, r in enumerate(ranked, start=1):
        if any(is_hit(r, g) for g in gold):
            return 1.0 / rank
    return 0.0
```

</details>

### What the BM25 run does show (37 questions, 1092 chunks per language)

- Where the first gold hit lands: rank 1 for 22 questions, ranks 2 to 5 for 5, ranks 6 to 10 for 3, ranks 11 to 50 for 6, outside the top 50 for 1.
- By question language, recall@10 / MRR: German 0.625 / 0.578 (n=12), English 0.846 / 0.610 (n=13), French 0.917 / 0.772 (n=12). By scope: AI Act 0.804 / 0.672 (n=23), GDPR 0.900 / 0.663 (n=10), questions needing both regulations 0.500 / 0.514 (n=4). With 4 to 13 questions per cell, differences of this size could be noise.
- **Most misses are vocabulary mismatches**, checked against the texts: q38 asks for "Bußgelder" but GDPR Art. 83 (German) says "Geldbußen" and never "Bußgeld" (not in the top 50); q04 says "banned" while Art. 99 says "prohibition"; q03 says "chatbot", which Art. 50 never uses ("interact directly with natural persons"); q34 uses the verb "beschweren" while Art. 85 has "Beschwerde", and my crude stemmer maps them to different tokens (`beschwer` vs `beschwerd`). q28 is not explained: Art. 16 is titled almost like the question and still ranks 32nd, and I did not isolate why.
- Known weakness of the tokenizer, seen but not fixed: "practices" becomes `practic` while "practice" stays `practice`. Fixing it after seeing the misses would be tuning on the test set, so any such change should be a separate, documented commit, ideally judged on questions not in this set.
- Hypothesis, **not a result**: dense embeddings should help exactly on these paraphrase misses, and hybrid should keep BM25's exact-term wins. The vector runs will confirm or refute that.

### Questions and answers

**1. Why does reciprocal rank fusion use ranks and not scores, and what does `k` do?**
A BM25 score and a cosine similarity are on different scales (BM25 is unbounded and depends on the query length and the corpus, cosine sits between -1 and 1), so adding or averaging them would let one ranker dominate by accident. Ranks are comparable across rankers by construction. `1 / (k + rank)` rewards a high place in any list, and a document found by both rankers gets two contributions. `k` controls how steep the reward is: with `k = 1` rank 1 is worth 1/2 and rank 2 is worth 1/3 (big gap), with `k = 60` they are worth 1/61 and 1/62 (almost the same), so the fused order depends on how many lists agree and not on tiny rank differences. A document in only one list is not discarded, it just scores less.

**2. Why does `anchor_id` exist, and what would have happened to q15 without it?**
q15 ("how does the regulation define an AI system?") has gold `aiact:art:3`. The corpus splits that article into `definition` units with ids like `aiact:def:1`, and the matching rule (equal to the gold id, or starting with it plus `:`) cannot match them. The whole article is not indexed as a chunk (it holds 68 definitions in one block), so nothing retrievable could hit the gold id and recall for q15 would be 0 for every method however good it is. `anchor_id` maps a definition to its parent article before matching. It is the only deviation from the README rule, documented in `evals/README.md` and worth your explicit approval, because it changes what counts as a hit.

**3. What failed on the first live run, why did waiting not help, and what did the cache buy us?**
Gemini's free tier has a per-day cap (1000 embedding texts), which is a different limit from the per-minute rate limit the router's backoff is built for. My first embedder retried for about 20 minutes against a limit that resets in about 16 hours. Now it detects the `PerDay` error and stops immediately (`QuotaExhaustedError`, with a test). The disk cache means nothing is embedded twice: the 1084 English embeddings, the 5 query embeddings done so far, and later every rerank score survive the failure, so each daily rerun continues where the last stopped and costs nothing for what is already done. The lesson is to check quotas and plan the cost before a long job, and to make jobs resumable.

---

## Stage 3: answers, API and chat page

### TODO items 6 and 7 (implemented)

| # | File | What it does |
|---|---|---|
| 6 | `src/copilot/answer/citations.py` | `parse_citation(label) -> cite id or None`: `"AI Act Art. 6(2)"` becomes `"aiact:art:6:2"` |
| 7 | `src/copilot/answer/refusal.py` | `should_refuse(evidence) -> bool`: refuse before any model call when retrieval is empty or weak |

*Status: both implemented. `test_every_corpus_label_round_trips` turns every one of the 1092 indexed chunks into its label and back through the parser.*

**Why #7 does not use BM25 scores: BM25 scores cannot tell out-of-scope questions from in-scope ones.** `make calibrate` (no API calls) prints, per gold question, the best BM25 score per query token: in-scope questions (n=37) range from 1.04 to 4.09 (median 1.77); the 3 out-of-scope questions score 0.00, 2.50 and 2.69, right inside that range, because "What is the best GPU for training a large language model?" shares words like "training" and "model" with the AI Act. Cosine similarity of the best vector hit is the better signal, and `refusal.py` uses it for English: `COSINE_MIN = {"en": 0.65}`, set between the lowest of the 9 in-scope English questions whose embeddings were fetched (0.699 to 0.814) and the one out-of-scope question (the GPU question, 0.599). **That is a provisional value resting on 1 negative against 9 positives**, because the bulk embedding quota ran out (it also blocked the 12 extra out-of-scope questions in `evals/refusal_v0.jsonl`). It is deliberately conservative since the model's `NOT_COVERED` answer and the grounding check still follow. French and German are not gated until their vectors exist and the threshold is recalibrated (`python -m copilot.answer.calibrate --embed`).

<details>
<summary>The implementations (now in <code>citations.py</code> and <code>refusal.py</code>)</summary>

```python
def parse_citation(label):
    label = " ".join(label.split())
    if match := _ARTICLE.match(label):  # "(AI Act|GDPR) Art. N(a)(b)..."
        regulation = _REGULATION[re.sub(r"\s+", " ", match.group(1).lower())]
        parts = re.findall(r"\(\s*(\w+)\s*\)", match.group(3))
        return ":".join([regulation, "art", match.group(2).lower(), *[p.lower() for p in parts]])
    if match := _ANNEX.match(label):  # "(AI Act|GDPR) Annex III[, Section A][, point 1]"
        ...  # same idea: regulation, "anx", numeral, optional section and point
    return None


def should_refuse(evidence):
    if evidence.n_sources == 0:
        return True
    if COSINE_MIN is not None and evidence.top_cosine is not None:
        return evidence.top_cosine < COSINE_MIN
    return False  # bm25 mode, or gate not calibrated: later safeguards decide
```

</details>

### Four safeguards, not one

A wrong answer can get through at four places, so there are four layers:

1. **Empty or weak retrieval** (`should_refuse`): no model call at all.
2. **The model's own `NOT_COVERED`**: the prompt tells it to answer exactly that when the sources do not answer the question. This is what refused the GPU question in the first live test, while BM25 alone would have let it through.
3. **Citation validation**: a citation that is unparsable or not among the supplied sources removes its sentence without asking anyone.
4. **The grounding check**: a second cheap model call judges every remaining sentence against the sources it cites; unsupported ones are removed, listed in `removed_sentences`, and `grounded` becomes false. If nothing survives, the answer is a refusal. If the checker itself is down, nothing can be verified: sentences with a valid citation are kept, the rest are removed, and `grounding_check` says `unavailable`.

### What the real runs taught us (all measured on 2026-10-29)

- **Model names go stale, and a model list lies.** `gemini-2.5-flash` and `-flash-lite` were in the list the API returned but answered `404 no longer available to new users`. The defaults are now `gemini-3.6-flash` (answers) and `gemini-3.1-flash-lite` (cheap checks), each tested with a real call. `gemini-3.8-flash`, the one the error message recommended, answered `503 high demand` or hung.
- **This is your Stage 0 TODO in the wild.** That 404 became a `BadRequestError`, and the placeholder `should_fallback` returns False for it, so the request failed even though Groq was configured and could have answered. Whether a "model not found" belongs with "bad request" (never fall back) or with "this provider cannot serve us" (fall back) is exactly the question the TODO asks.
- **Timeouts matter.** One test call to `gemini-3.8-flash` hung for minutes because the SDK had no useful default; both providers now have a 60 s timeout, which the router treats as a retryable `ServerError`.
- **Thinking time dominates latency.** On one prompt (a single test, noisy) the default Gemini setting took 7 to 22 s for about 210 output tokens; `thinking_level="MINIMAL"` gave the same answer in 2.7 s. The answer model uses minimal thinking because it extracts from supplied sources and does not need to reason. Streaming shows the first text early either way.
- **Groq's free tier caps its large model at 8000 tokens per minute**, which a few of our 2,000-token prompts per minute exceed. The generation chain is now the lite Gemini model first (it answered every test call in 1 to 2 s), the larger Gemini model second, and Groq last, with 30 s timeouts and 3 s backoff so a "try again in 5 s" limit is actually waited out. Which models answered is reported in `usage.models`.
- **A source cut off silently loses content.** My first limit of 1,500 characters per source dropped five of the eight prohibited practices in Art. 5(1) (4,701 characters). It is 6,000 now.

- **Latency is mostly provider load, not our pipeline.** In the 10 README examples (after the reorder) per-question time was 1.7 to 70 s: five took 4.3 s or less and five took 15 s or more. Asked again minutes later, three of the slow ones took 3.4, 4.1 and 3.4 s (about 2 s generating, 1.5 s checking). I did not isolate the cause of the slow runs; 503 "high demand" and Groq rate-limit waits are the candidates. In the first run, before the reorder, examples took 10 to 87 s.
- **"Grounded" is not "correct" or "complete".** Example 7 (an employer detecting emotions) is grounded sentence by sentence and cites real articles, yet never states the actual prohibition in Art. 5(1)(f), because BM25 did not retrieve it. The check verifies faithfulness to the sources given, not that the right sources were found. Two examples (the bank loan and the French definition of an AI system) were refused because retrieval missed the right articles, the same misses the Stage 2 BM25 run showed.

### Questions and answers

**1. Why does the system have four layers that can refuse or remove text, and why is a BM25 threshold not one of them?**
Each layer catches a different failure: retrieval found nothing or something unrelated (cheap, no model call), retrieval found plausible but non-answering text (the model says `NOT_COVERED`), the model cited something it was not given (validation), or the model stated more than the sources say (grounding check). A BM25 threshold does not work because the score depends on the query length and the corpus statistics, and measured on the gold set the out-of-scope questions score inside the in-scope range (per token: 2.50 and 2.69 against 1.04 to 4.09). Cosine similarity has a fixed scale, which is why that gate exists, but it is not calibrated yet.

**2. Why is a citation to "Art. 6" accepted when the source is "Art. 6(2)", but "Art. 6(1)" is rejected, and what does the rule protect against?**
`supports(cited, source)` accepts equal ids or one containing the other (an ancestor or a descendant). Citing the article when the model was shown one of its paragraphs is a fair summary, and citing a point below a paragraph is fair too, so both are still checked against the source text by the grounding call. Siblings are rejected because a different paragraph was never shown to the model, so a sentence citing it cannot have come from what was supplied: the model is fabricating a plausible-looking reference, which is the most damaging error for a legal assistant. Such sentences are dropped without asking the checker.

**3. Why is the streamed text only a draft, and why does the router fall back only before the first chunk?**
The grounding check needs the whole answer, so it runs after streaming, and it may remove sentences. The page therefore shows the stream as grey italic text and replaces it with the `final` event, the only verified version. The router can retry or switch provider only until the first event arrives, because after that the user has already seen text: switching mid-answer would repeat or contradict it, so an error at that point is raised and the draft is discarded.

### Open items for tomorrow (the quota resets about 08:00; the evaluation task starts at 08:10)

1. Embed the remaining gold questions (single-text calls kept working after the bulk quota was spent, so this should be possible at any time; if the evaluation job's final query step fails on quota, that is why), then rerun `make calibrate` and `make examples`.
2. Set the refusal threshold from that data (TODO #7), and add out-of-scope questions to the gold set.
3. When the evaluation finishes (about three days), check which retrieval method is actually best before treating hybrid as the default.

---

## Stage 4A: agent design (no code)

The design is in `docs/agent-design.md`. Owner decisions (2026-10-29): LangGraph for the graph only (models through our provider layer), `classify_ai_risk` = the model extracts facts and rules decide, a stateless API for now, and an indicative compliance report that lists its assumptions and missing facts.

### Questions and answers

**1. Why is the compliance check a fixed plan, while ordinary questions get a search loop?**
For a compliance check the structure is known before starting: classify the system, list the obligations for each role, check the GDPR overlap, report. A fixed plan can be shown to the user before it runs, tested step by step, and checked in traces ("did the run visit every step?"). For an ordinary question we do not know in advance which provisions are needed or whether the first search will find them, so a short loop (rewrite, search, grade, at most 2 re-searches) adapts to what retrieval returns. The general rule from `design.md`: a workflow wherever the steps are known, an agent loop only where they are not, and even then a capped one.

**2. Why does `classify_ai_risk` let rules decide the category instead of the model?**
Because the category is the most consequential output and has to be auditable. The model only does what it is good at, turning a free-text description into a fixed checklist of facts, each tied to a phrase of the description. The decision is then a deterministic rule table over Art. 5, Art. 6 and Annex III: the same facts always give the same category, every row has a test case, and the table can be reviewed against the text. A fact the description does not settle stays "unknown", which makes the result "undetermined" and turns into a question for the user, instead of the model filling the gap with a plausible guess. The cost is coverage: a situation the rules do not foresee gets "undetermined", which is the honest outcome for an indicative tool.

**3. What happens when a run hits its token budget in the middle, and why does the budget guard keep some budget back?**
The run does not crash and does not return an unchecked answer. It ends with `status = "partial"` and `limit_hit = "tokens"`: if a grounded draft exists it is returned with what it covers, a compliance run returns its completed steps and missing facts, and the answer says plainly that it is partial. The guard reserves one model call and about 6,000 tokens for the grounding check, because the most dangerous output is a fluent draft that nobody verified; without the reserve, a run that spent everything on searching would have to choose between showing an unchecked draft and showing nothing. With it, the last draft can still be checked, and if even that is impossible the draft is withheld and a refusal is returned.
