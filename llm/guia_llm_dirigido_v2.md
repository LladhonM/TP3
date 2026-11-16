# Guideline for the steered LLM classifier

Version: **v2**. Previous versions are kept as `guia_llm_dirigido_vN.md`.
Each steering round adds rules to the "Human rules" section and bumps the version; every rule is checked first on train rows
outside the steering and validation samples. Once frozen, this file is applied unchanged to the validation lots and, only if
validation passes, to the test band.

---

## Task

Each row is a MercadoLibre Argentina listing from 2015. Estimate the probability that the item is **used** (`usado`) rather than
**new** (`nuevo`).

These rows were chosen because a strong model (gradient-boosted trees + text models + seller statistics) is close to 50/50 on
them. Roughly **47% of rows like these are used**, so do not lean toward either class by default. Your value is world
knowledge about the specific product, which a word-count model does not have.

## Columns

- `titulo`: listing title (Spanish).
- `garantia`: warranty text written by the seller; `sin_dato` = empty.
- `atributos`: structured attributes (first 150 characters), often empty.
- `tipo_publicacion`: listing tier (`free`, `bronze`, `silver`, `gold`, `gold_special`, `gold_premium`, `gold_pro`).
- `precio`: price in ARS (2015; roughly 9–15 ARS per USD).
- `stock_inicial`: initial quantity offered. `vendidos`: units sold.
- `n_fotos`: number of photos. `pct_fotos_camara`: share of photos with a 4:3 aspect ratio (phone or camera shots).
- `publicaciones_vendedor_train`: how many listings this seller has in the dataset.

## How to judge

- Read each row yourself and judge it individually. Do not write code to score rows.
- **The seller signals carry almost no information in these rows.** Measured on similar rows: free tier 48% used, bronze 47%,
  stock 1 47%, phone-style photos 47%, sellers with 1–2 listings 47% — all equal to the 47% average. The model already used
  them; that is why these rows are ambiguous. Do not move your probability because of tier, stock, photos or seller size.
- **The product category alone is not decisive either.** In these rows books are 46% used, clothing 46%, music/films/games 55%,
  furniture and decor 29–43%. Do not lean used just because something is clothing or furniture.
- Decide on what the text says about the **specific product**: the exact model, edition, publisher, condition words, whether it
  is the kind of item a shop stocks or a person resells, and anything your world knowledge says about it.
- Give a calibrated probability between 0.02 and 0.98. Use values near 0.5 when the text gives you nothing specific.

## Human rules

These were proposed by a human reviewer or found in train data, and checked on train rows outside this sample. They are soft leans, not certainties:
move the probability to about 0.35–0.40 or 0.60–0.65, unless the listing has stronger evidence of its own.

1. **Books.** Literature, essays, history and general non-fiction lean slightly **new** (about 45% used). Study material —
   school or university textbooks, exam preparation, manuals, course books — leans **used** (about 70% used, small sample).
2. **Phones.** A named, recent phone model (e.g. Samsung Galaxy S6, Grand Prime, Moto G, iPhone) with no mention of defects,
   parts or accessories leans **new** (about 39% used).
3. **Professional audio and video gear** — studio monitors, mixers, consoles, amplifiers, pedals, microphones, audio
   interfaces, lenses, tripods, stabilizers — sold by a seller with 1–2 listings leans **used** (about 65% used). This is
   the one case where seller size matters.
4. **Condition words describe a used item.** "sin uso", "como nuevo", "impecable", "excelente estado", "muy buen estado",
   "buen estado", "poco uso" lean **used** (about 65% used). People write these when reselling something they owned; a shop
   selling new stock does not describe the condition. Do not read "sin uso" or "como nuevo" as new.
5. **"Nuevo" / "nueva" in the title is weaker than it looks.** In these rows it is only about 59% new. Lean new (about
   0.40), not strongly new, unless the listing adds more evidence such as "sellado", "en caja cerrada" or a shop-style
   warranty.
