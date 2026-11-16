# Prompt for the steered LLM classifier (Plan C, models #49)

Run with one Sonnet subagent per lot of up to 100 rows. Replace `<VERSION>`, `<LOT>`, `<FIRST_ID>`–`<LAST_ID>` and `<N>`.
Lots and keys are built by `muestra_llm_dirigido.py` (train: A–C steering, V1–V4 validation) and `lotes_llm_test.py` (test: T01–T15).
Outputs go to `data/processed/llm_dirigido/resultados/<VERSION>_<LOT>.json` and are read by notebook 49.

During steering the guideline path was `notebooks/_generadores/guia_llm_dirigido.md` (then at v0, v1, v2). For validation and test the
frozen copy `notebooks/_generadores/guia_llm_dirigido_v2.md` was used (identical bytes to the v2 steering run). The final message asked
for doubt themes during steering and only a confirmation for test lots; the per-row instructions were identical.

---

You are classifying MercadoLibre listings as new or used, as part of a data-science competition. Work in English.

Project root: <PROJECT ROOT>

1. Read the guideline: notebooks/_generadores/guia_llm_dirigido_v2.md (version v2). Follow it exactly. Do not open any other version of it.
2. Read the <N> rows in: data/processed/llm_dirigido/lote_<LOT>.csv
3. Do NOT open, list or search any other file or folder in the project (in particular nothing else under data/). The labels must stay
   hidden from you; opening anything else invalidates the experiment.

For EVERY row (ids <FIRST_ID>–<LAST_ID>), judge it yourself using the guideline and give p = probability the item is USED (0.02–0.98).
Do not write code that scores rows; read the rows and decide. You may use code only to read the CSV and to write the output file.

For rows where you are genuinely unsure (p between 0.35 and 0.65) or where your answer rests on an assumption you are not confident about,
also write a short doubt, phrased as a GENERAL question a human expert could answer once and that would apply to many listings. Leave doubt
empty otherwise.

Write the output as JSON to: data/processed/llm_dirigido/resultados/<VERSION>_<LOT>.json
Format: {"<FIRST_ID>": {"p": 0.70, "duda": ""}, ...} with all <N> ids.

Your final message: confirm the file was written and how many ids it has (steering rounds: also the main doubt themes as general
questions with their row ids). Do not report per-row predictions.
