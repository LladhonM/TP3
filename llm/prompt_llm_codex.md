You are classifying MercadoLibre listings as new or used, as part of a data-science competition. Work in English.

Your working folder contains exactly two input files. Use only these:

1. Read the guideline: guia.md (version v2). Follow it exactly.
2. Read the <N> rows in: lote.csv
3. Do NOT open, list or search any other file or folder on this machine. The labels must stay hidden from you; opening anything
   else invalidates the experiment.

For EVERY row (ids <FIRST_ID>–<LAST_ID>), judge it yourself using the guideline and give p = probability the item is USED (0.02–0.98).
Do not write code that scores rows; read the rows and decide. You may use code only to read the CSV and to write the output file.

For rows where you are genuinely unsure (p between 0.35 and 0.65) or where your answer rests on an assumption you are not confident about,
also write a short doubt, phrased as a GENERAL question a human expert could answer once and that would apply to many listings. Leave doubt
empty otherwise.

Write the output as JSON to: resultado.json (in your working folder)
Format: {"<FIRST_ID>": {"p": 0.70, "duda": ""}, ...} with all <N> ids.

Your final message: confirm the file was written and how many ids it has. Do not report per-row predictions.
