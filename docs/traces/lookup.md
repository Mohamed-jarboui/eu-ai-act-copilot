# Trace: lookup

Captured 2026-10-30 with `python -m copilot.agent.traces` (real models, unedited). Raw events: the `.json` file next to this one.

**Question (fr):** Que dit l'article 22 du RGPD ?

**Outcome:** route `lookup`, status `done`, grounded `True`, retrieval `None`, limit hit `None`.

**Usage:** 30.0 s, 6 steps, 2 model calls (gemini:gemini-3.1-flash-lite, groq:openai/gpt-oss-20b), 1 tool calls, 1522 input / 321 output tokens.

## Reasoning steps

1. Route: **lookup** (confidence 1.0, rules) -> GDPR Art. 22
2. Tool `get_article` {"regulation": "gdpr", "reference": "22", "language": "fr"}
3.   -> GDPR Art. 22 (0 ms)
4. Model for answer: gemini:gemini-3.1-flash-lite (17683 ms)
5. Draft: 170 words from GDPR Art. 22
6. Model for ground: groq:openai/gpt-oss-20b (12261 ms)
7. Grounding check: kept 4, removed 0 (round 0)

## Answer

> La personne concernée a le droit de ne pas faire l'objet d'une décision fondée exclusivement sur un traitement automatisé, y compris le profilage, lorsqu'elle produit des effets juridiques la concernant ou l'affectant de manière significative [GDPR Art. 22]. Cette règle ne s'applique pas si la décision est nécessaire à la conclusion ou à l'exécution d'un contrat, si elle est autorisée par le droit de l'Union ou d'un État membre prévoyant des mesures de sauvegarde, ou si elle repose sur le consentement explicite de la personne [GDPR Art. 22]. Dans les cas où la décision est nécessaire à un contrat ou fondée sur le consentement, le responsable du traitement doit mettre en œuvre des mesures appropriées, incluant au minimum le droit d'obtenir une intervention humaine, d'exprimer son point de vue et de contester la décision [GDPR Art. 22]. Enfin, ces décisions ne peuvent pas être fondées sur des catégories particulières de données, sauf si des exceptions spécifiques s'appliquent et que des mesures de sauvegarde appropriées sont en place [GDPR Art. 22].

## Citations

- GDPR Art. 22: 1. La personne concernée a le droit de ne pas faire l'objet d'une décision fondée exclusivement sur un traitement automatisé, y compris le profilage, produisant
