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

---

## Stage 2: retrieval

### Status: the vector numbers are not in yet

The first live run embedded all English chunks (1084 distinct texts) and then hit Gemini's free-tier **daily** limit (1000 embedding requests per day, "retry in about 16h"). The only results so far are BM25 on all 37 scored questions (`results/retrieval_bm25_all.json`). To finish, rerun the commands in `evals/README.md` ("Finishing the vector results") once per day, or enable billing on the key (your call). Until then there is no honest answer to "which method wins".

### TODO #3, #4, #5: `rrf_fuse`, `recall_at_k`, `reciprocal_rank`

| # | File | Formula |
|---|---|---|
| 3 | `src/copilot/retrieve/fusion.py` | `score(d) = sum over rankings r containing d of 1 / (k + rank_r(d))`, rank starting at 1, `k = 60`; sort by score, ties in first-seen order |
| 4 | `src/copilot/retrieve/metrics.py` | `recall@k = |{g in gold : some retrieved r in the top k has is_hit(r, g)}| / |gold|` |
| 5 | same | `RR = 1 / rank of the first hit` (0 if none); `MRR = mean(RR)` |

How to know you're done: the `mine` cases in `tests/test_retrieve_fusion_metrics.py` stop being xfail (remove the markers), then set `IMPL = "mine"` in `evals/run_retrieval.py` and rerun. **The numbers must not change.** You can already check #4 and #5 against the BM25 run, which used the reference implementation on 37 questions:

| Method | recall@5 | recall@10 | MRR |
|---|---|---|---|
| bm25 | 0.694 | 0.797 | 0.652 |

(`python evals/run_retrieval.py --methods bm25 --tag bm25_all` with your implementation should print exactly these.) #3 can only be checked against the vector runs, but its tests have exact expected scores.

<details>
<summary>Reference implementations (they live in <code>src/copilot/retrieve/_reference.py</code>; try yours first)</summary>

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
