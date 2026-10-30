# EU AI Act Copilot

Ask questions about the EU AI Act and GDPR in English, French or German and get answers grounded in the official texts, with article-level citations. An agent can also run compliance checks for a described AI system.

**Status:** in progress. Design notes: [docs/design.md](docs/design.md).

## How to run

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is installed by uv if missing.

```bash
make install            # uv sync
cp .env.example .env    # then add GEMINI_API_KEY and/or GROQ_API_KEY
make lint               # ruff check + format check
make test               # pytest (no network, fake providers)
make ingest             # parse data/raw into data/processed/corpus.jsonl (downloads missing files)
make eval-retrieval     # recall@k / MRR for BM25, vector, hybrid, hybrid+rerank (needs both API keys)
make serve              # API and chat page on http://127.0.0.1:8000 (needs both API keys)
make examples           # rerun the example questions below and rewrite that section
make calibrate          # retrieval scores behind the refusal decision (no API calls)
make eval-router        # router accuracy on the route-labelled questions (needs the API keys)
make traces             # capture one real agent run per graph path into docs/traces/
```

Corpus format, sources, licence and limitations: [docs/data-card.md](docs/data-card.md).

To run a script with your keys loaded: `uv run --env-file .env python your_script.py`.

## Using it

`make serve`, then open http://127.0.0.1:8000. The same pipeline is available over HTTP:

- `POST /ask` with `{"question": "...", "language": "en|fr|de"}` (language optional, detected if omitted) returns the answer, `citations` (id, label, snippet), `grounded`, `retrieval_mode`, `removed_sentences` and `usage` (tokens, models used, latency per stage).
- `POST /ask/stream` sends server-sent events: `sources`, draft `delta` text, then one `final` event with the verified answer. Only `final` has been checked against the sources.
- `GET /health` reports, per language, whether requests use `hybrid` (BM25 + vectors) or `bm25` only.

How an answer is made: retrieve the best chunks in the question's language, refuse if retrieval is empty or weak, let a model answer using only those numbered sources with `[AI Act Art. 6(2)]`-style citations (or `NOT_COVERED`), then a second cheap model call checks every sentence against the sources it cites. Unsupported sentences are removed and listed; if nothing is left the system says the texts do not cover the question. Citations that point outside the supplied sources are removed without asking the checker.

Known limits: the daily Gemini free-tier embedding quota means vectors exist for English only for now (other languages answer with BM25, which the evaluation shows misses paraphrases); the cosine refusal threshold exists for English only and is provisional (9 in-scope and 1 out-of-scope question, see `docs/learning-log.md`); "grounded" means the sentences are supported by the sources that were retrieved, not that the right sources were found (see example 7); response times depend heavily on provider load (the examples below took 1.7 to 70 s, and three of the slow ones took about 4 s when asked again); this gives information about the texts, not legal advice.

## The agent

`POST /agent` (and the **Agent** mode of the chat page, with a **Show reasoning steps** panel) runs a LangGraph agent (`src/copilot/agent/`, design in [docs/agent-design.md](docs/agent-design.md)):

- an input guard, then a **router**: explicit references ("Article 22 GDPR") are routed by rules without a model call; otherwise a small model chooses `lookup`, `legal_question`, `compare`, `compliance_check` or `out_of_scope`, and low confidence asks one clarifying question;
- questions go through an **agentic RAG loop** (rewrite, hybrid search, grade, at most 2 re-searches, answer); a described AI system goes through a **plan-and-execute compliance check** (classify the risk level with deterministic rules over extracted facts, list obligations per role, check the GDPR overlap, report with assumptions and missing facts);
- six **read-only tools** with typed arguments and recovery-oriented errors, per-step allow-lists, and guards that the category comes from the classifier and that the classifier sees the user's own words;
- per-route **limits** (steps, model calls, tool calls, tokens, time), repeated-call detection, and a partial, flagged result instead of a crash; the Stage 3 **grounding check** runs last, with a reserve kept for it;
- conversations persist per `thread_id` in SQLite, so the answer to a clarifying question continues the same thread.

The stream sends one `trace` event per step (routing decision, plan, tool calls and results, grading, critic, limits), then one `final` event. Three real runs, one per graph path, are in [docs/traces/](docs/traces/).

Router accuracy (`evals/run_router.py`, 74 route-labelled questions drafted by me and still to be reviewed): 0.905, or 0.943 on the 70 not marked ambiguous; all 7 errors sent a legal question to the compliance check, and an earlier identical run scored 70 of 74, so routing varies between runs.

## Example questions

<!-- examples:start -->
Real output of `python -m copilot.answer.examples`, unedited. Vectors were available for: en; every other language, and any question whose embedding could not be fetched, used BM25 only (see each `retrieval:`).

**1. [EN] Which uses of AI are completely banned in the EU, whoever deploys them?**

> The AI Act expressly forbids the placing on the market, the putting into service or the use of any AI system that (a) employs subliminal or purposefully manipulative or deceptive techniques that materially distort a person’s or group’s behaviour and are likely to cause significant harm; (b) exploits vulnerabilities of a natural person or a specific group because of age, disability or a particular social or economic situation with the aim of materially distorting behaviour and causing significant harm; (c) evaluates or classifies natural persons or groups over time on the basis of their social behaviour or inferred personal characteristics, where the resulting social score leads to detrimental or unfavourable treatment that is unrelated to the original context or is unjustified or disproportionate; (d) makes risk assessments of natural persons to predict the risk of committing a criminal offence solely on profiling or personality traits, except when used only to support a human assessment based on objective, verifiable facts; (e) creates or expands facial‑recognition databases through untargeted scraping of facial images from the internet or CCTV footage; (f) infers emotions of a natural person in workplaces or educational institutions, unless the system is intended for medical or safety reasons; (g) categorises individuals on biometric data to infer race, political opinions, trade‑union membership, religious or philosophical beliefs, sex life or sexual orientation, without covering lawful labelling or filtering of already acquired biometric datasets for law‑enforcement purposes; and (h) uses real‑time remote biometric identification systems in publicly accessible spaces for law‑enforcement purposes, except where strictly necessary for the targeted search of victims of abduction, trafficking or sexual exploitation, the prevention of a substantial imminent threat to life or safety or a genuine terrorist threat, or the localisation or identification of a suspect for a serious criminal investigation or prosecution [AI Act Art. 5(1)].

Citations: AI Act Art. 5(1)  
Status: grounded · retrieval: hybrid · 32.6 s · 4324 in / 898 out tokens · models: groq:openai/gpt-oss-120b, groq:openai/gpt-oss-20b

**2. [EN] Is an AI tool that ranks job applicants considered high-risk?**

> AI systems intended to be used for the recruitment or selection of natural persons, including those used to analyse and filter job applications and evaluate candidates, are listed in Annex III [AI Act Annex III, point 4]. Generally, AI systems referred to in Annex III are considered to be high-risk [AI Act Art. 6(2)]. However, an AI system referred to in Annex III shall not be considered high-risk if it does not pose a significant risk of harm to the health, safety, or fundamental rights of natural persons, such as by not materially influencing the outcome of decision-making [AI Act Art. 6(3)]. This exception applies if the system performs a narrow procedural task, improves the result of a previously completed human activity, detects patterns without replacing or influencing human assessment, or performs a preparatory task [AI Act Art. 6(3)]. Regardless of these conditions, an AI system referred to in Annex III is always considered high-risk if it performs profiling of natural persons [AI Act Art. 6(3)]. If a provider determines that an Annex III system is not high-risk, they must document this assessment before the system is placed on the market or put into service [AI Act Art. 6(4)].

Citations: AI Act Annex III, point 4, AI Act Art. 6(2), AI Act Art. 6(3), AI Act Art. 6(4)  
Status: grounded · retrieval: hybrid · 3.4 s · 2340 in / 724 out tokens · models: gemini:gemini-3.1-flash-lite, groq:openai/gpt-oss-20b

**3. [EN] Within how many hours must a data breach be reported to the authority?**

> In the event of a personal data breach, the controller must notify the competent supervisory authority without undue delay and, where feasible, no later than 72 hours after becoming aware of it [GDPR Art. 33(1)]. If the notification is not made within this 72-hour period, it must be accompanied by reasons for the delay [GDPR Art. 33(1)].

Citations: GDPR Art. 33(1)  
Status: grounded · retrieval: hybrid · 19.3 s · 1411 in / 198 out tokens · models: gemini:gemini-3.1-flash-lite, groq:openai/gpt-oss-20b

**4. [EN] A bank's AI model refuses loans automatically. What can the rejected applicant demand?**

> The texts do not cover this question.

Citations: none  
Status: no answer (not_covered) · retrieval: hybrid · 3.1 s · 1306 in / 4 out tokens · models: gemini:gemini-3.1-flash-lite

**5. [EN] What is the best GPU for training a large language model?**

> The texts do not cover this question.

Citations: none  
Status: no answer (not_covered) · retrieval: hybrid · 1.7 s · 1663 in / 4 out tokens · models: gemini:gemini-3.1-flash-lite

**6. [FR] Comment le règlement définit-il un système d'IA ?**

> Les textes ne couvrent pas cette question.

Citations: none  
Status: no answer (not_covered) · retrieval: bm25 · 2.4 s · 1541 in / 4 out tokens · models: gemini:gemini-3.1-flash-lite

**7. [FR] Un employeur peut-il utiliser l'IA pour détecter les émotions de ses salariés ?**

> Un système de reconnaissance des émotions est défini comme un système d'IA permettant la déduction des émotions ou des intentions de personnes physiques sur la base de leurs données biométriques [AI Act Art. 3(39)]. Lorsqu'un employeur utilise un tel système, il doit informer les représentants des travailleurs et les travailleurs concernés qu'ils seront soumis à l'utilisation du système d'IA à haut risque avant sa mise en service [AI Act Art. 26(7)]. Par ailleurs, les déployeurs d'un système de reconnaissance des émotions ont l'obligation d'informer les personnes physiques qui y sont exposées du fonctionnement du système et de traiter les données à caractère personnel conformément au règlement (UE) 2016/679 [AI Act Art. 50(3)]. Le règlement n'empêche pas les États membres de maintenir ou d'introduire des dispositions législatives plus favorables aux travailleurs concernant la protection de leurs droits face à l'utilisation de systèmes d'IA par les employeurs [AI Act Art. 2(11)]. Enfin, les systèmes d'IA à haut risque doivent être fournis de manière à permettre un contrôle humain, incluant la capacité d'interpréter les sorties du système et de décider de ne pas les utiliser ou de les ignorer dans une situation particulière [AI Act Art. 14(4)].

Citations: AI Act Art. 3(39), AI Act Art. 26(7), AI Act Art. 50(3), AI Act Art. 2(11), AI Act Art. 14(4)  
Status: NOT fully grounded (removed_sentences) · retrieval: bm25 · 4.3 s · 3094 in / 516 out tokens · models: gemini:gemini-3.1-flash-lite, groq:openai/gpt-oss-20b
  
Removed as unsupported: Ces systèmes sont classés comme des systèmes d'IA à haut risque lorsqu'ils sont utilisés dans les cas prévus par la réglementation [AI Act Annex III, point 1].

**8. [FR] Quand un consentement est-il valable ?**

> Il doit être présenté séparément de toute autre déclaration, sous une forme clairement distincte, compréhensible, aisément accessible et rédigée en termes simples [GDPR Art. 7(2)]. Lors de l’évaluation de la liberté du consentement, il faut veiller à ce que l’exécution d’un contrat ou la fourniture d’un service ne soit pas conditionnée à un consentement à un traitement qui n’est pas nécessaire à ce contrat [GDPR Art. 7(4)]. La personne concernée peut retirer son consentement à tout moment, le retrait étant aussi simple que le consentement initial et n’affectant pas la licéité du traitement réalisé avant ce retrait [GDPR Art. 7(3)]. Dans le cas des enfants, le responsable du traitement doit vérifier de façon raisonnable que le consentement provient ou est autorisé par le titulaire de la responsabilité parentale, en fonction des moyens technologiques disponibles [GDPR Art. 8(2)].

Citations: GDPR Art. 7(2), GDPR Art. 7(4), GDPR Art. 7(3), GDPR Art. 8(2)  
Status: NOT fully grounded (removed_sentences) · retrieval: bm25 · 70.0 s · 2041 in / 1047 out tokens · models: groq:openai/gpt-oss-120b, groq:openai/gpt-oss-20b
  
Removed as unsupported: Le consentement est valable lorsqu’il est donné librement, de façon spécifique, éclairée et univoque, et que le responsable du traitement est capable de le démontrer [GDPR Art. 7(1)].

**9. [DE] Müssen Deepfakes gekennzeichnet werden?**

> Betreiber eines KI-Systems, das Bild-, Ton- oder Videoinhalte erzeugt oder manipuliert, die ein Deepfake darstellen, müssen offenlegen, dass diese Inhalte künstlich erzeugt oder manipuliert wurden [AI Act Art. 50(4)]. Handelt es sich bei dem Inhalt um ein offensichtlich künstlerisches, kreatives, satirisches, fiktionales oder analoges Werk oder Programm, beschränkt sich diese Pflicht darauf, das Vorhandensein solcher Inhalte in geeigneter Weise offenzulegen, ohne die Darstellung oder den Genuss des Werks zu beeinträchtigen [AI Act Art. 50(4)]. Diese Offenlegungspflicht gilt nicht, wenn die Verwendung zur Aufdeckung, Verhütung, Ermittlung oder Verfolgung von Straftaten gesetzlich zugelassen ist [AI Act Art. 50(4)]. Zusätzlich müssen Anbieter von KI-Systemen, die synthetische Audio-, Bild-, Video- oder Textinhalte erzeugen, sicherstellen, dass die Ausgaben des Systems in einem maschinenlesbaren Format gekennzeichnet und als künstlich erzeugt oder manipuliert erkennbar sind [AI Act Art. 50(2)]. Diese Pflicht gilt nicht, wenn die KI-Systeme eine unterstützende Funktion für die Standardbearbeitung ausführen, die Eingabedaten oder deren Semantik nicht wesentlich verändern oder wenn sie zur Aufdeckung, Verhütung, Ermittlung oder Verfolgung von Straftaten gesetzlich zugelassen sind [AI Act Art. 50(2)]. Ein Deepfake ist dabei als ein durch KI erzeugter oder manipulierter Inhalt definiert, der wirklichen Personen, Gegenständen, Orten, Einrichtungen oder Ereignissen ähnelt und einer Person fälschlicherweise als echt oder wahrheitsgemäß erscheinen würde [AI Act Art. 3(60)].

Citations: AI Act Art. 50(4), AI Act Art. 50(2), AI Act Art. 3(60)  
Status: grounded · retrieval: bm25 · 15.6 s · 2130 in / 480 out tokens · models: gemini:gemini-3.1-flash-lite, groq:openai/gpt-oss-20b

**10. [DE] Wann darf ich verlangen, dass meine Daten gelöscht werden?**

> Sie können die unverzügliche Löschung Ihrer personenbezogenen Daten verlangen, wenn diese für die Zwecke, für die sie erhoben oder verarbeitet wurden, nicht mehr notwendig sind [GDPR Art. 17(1)]. Zudem besteht dieses Recht, wenn Sie Ihre Einwilligung widerrufen und keine anderweitige Rechtsgrundlage für die Verarbeitung vorliegt, oder wenn Sie Widerspruch gegen die Verarbeitung eingelegt haben und keine vorrangigen berechtigten Gründe für die Verarbeitung bestehen [GDPR Art. 17(1)]. Auch bei einer unrechtmäßigen Verarbeitung, zur Erfüllung einer rechtlichen Verpflichtung oder bei einer Erhebung in Bezug auf Dienste der Informationsgesellschaft gemäß Artikel 8 Absatz 1 kann die Löschung verlangt werden [GDPR Art. 17(1)]. Im Kontext von Tests von Hochrisiko-KI-Systemen unter Realbedingungen können Sie Ihre Teilnahme jederzeit durch Widerruf der informierten Einwilligung beenden und die unverzügliche und dauerhafte Löschung Ihrer personenbezogenen Daten verlangen, ohne dass dies begründet werden muss [AI Act Art. 60(5)].

Citations: GDPR Art. 17(1), AI Act Art. 60(5)  
Status: grounded · retrieval: bm25 · 31.1 s · 2429 in / 610 out tokens · models: gemini:gemini-3.1-flash-lite, groq:openai/gpt-oss-20b
<!-- examples:end -->
