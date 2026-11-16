# Lecturas de LLM: cómo se hicieron y cómo replicarlas

El modelo final usa, en las filas donde la capa 1 duda, la probabilidad de que el producto sea usado según cuatro LLMs que
**leyeron cada publicación** con una guía escrita y congelada. Esas lecturas son un insumo fijo del modelo
(`data/llm/resultados/`). Esta carpeta tiene lo necesario para volver a generarlas.

**Aviso:** los LLM no son deterministas. Volver a correr las lecturas da valores parecidos, no idénticos. Por ejemplo, dos
corridas de Opus con la misma guía y los mismos lotes correlacionan 0.96. Si se reemplazan las lecturas, el modelo cambia y hay
que volver a correr los dos notebooks.

## Cómo se hizo

### 1. Qué filas se leyeron

No se leyó todo el dataset: solo las filas donde el stacking de árboles dudaba, en tres zonas definidas por su probabilidad de
usado `p`.

| Zona | Criterio | Train leído | Test leído |
|---|---|---|---|
| Banda | \|p − 0.45\| < 0.2 | 3.084 filas (lotes A–C, V1–V4, E01–E08, N01–N16) | 1.476 (T01–T15) |
| Anillo | 0.2 ≤ \|p − 0.45\| < 0.3 | 2.575 (R01–R08, RN01–RN17, RX01) | 1.027 (TR01–TR11) |
| Bolsillo | fuera de banda y anillo, con el modelo con TabPFN entre 0.2 y 0.8 | 800 (P01–P08) | 549 (TP01–TP06) |

Las muestras de train se sortearon con semilla fija dentro de cada zona. En test se leyeron todas las filas de cada zona. Las
zonas se definieron una sola vez, con el stacking de ese momento, y quedaron fijas: hoy son la lista de filas de
`data/llm/clave*.csv`.

### 2. Qué ve el LLM

Cada lote (`data/llm/lotes/lote_<LOTE>.csv`) tiene hasta 100 publicaciones con estas columnas: título, garantía, atributos, tipo
de publicación, precio, stock inicial, vendidos, cantidad de fotos, proporción de fotos de cámara (4:3) y cantidad de
publicaciones del vendedor.

- El `id` de cada fila es un número de lectura (10001, 10002, …), no el número de fila del dataset.
- **El LLM nunca ve la etiqueta ni la predicción de ningún modelo.** El mapeo `id → fila` y la etiqueta están en las claves, que
  el LLM no puede abrir.

### 3. La guía (`guia_llm_dirigido_v2.md`)

Se escribió en rondas, con revisión humana:

1. Un LLM leyó 300 filas de train de la banda (lotes A–C) y dejó, en las filas dudosas, preguntas generales para un experto.
2. Una persona revisó esas dudas y los errores, y propuso reglas. **Cada regla se probó primero en ~3.000 filas de la banda que
   no estaban en ninguna muestra**, y solo entró si se sostenía ahí.
3. La guía pasó por tres versiones (v0 → v1 → v2) y se congeló en la v2.

Lo que se aprendió y quedó escrito en la guía: dentro de estas filas, las señales del vendedor (tipo de publicación, stock,
fotos, tamaño del vendedor) **no informan**, porque el modelo ya las usó. Lo que suma es el conocimiento del producto concreto.
Hay 5 reglas suaves: libros, teléfonos, equipos de audio y video profesionales, palabras de estado ("impecable", "sin uso") y
"nuevo" en el título.

Con la guía congelada se hizo una validación ciega en 400 filas nuevas (V1–V4) y en 800 más (E01–E08). Después se aplicó sin
cambios a todos los demás lotes de train y de test.

### 4. Los cuatro LLM

Cada lote corrió en una carpeta temporal aislada que solo contenía `guia.md` y `lote.csv`, con permisos para leer esos dos
archivos y escribir el resultado. El modelo no veía el proyecto.

| Corrida | Modelo | Herramienta | Prompt | Script |
|---|---|---|---|---|
| `o2` | Claude Opus | subagentes de Claude Code | `prompt_llm_dirigido.md` | (orquestado desde Claude Code) |
| `sol` | GPT 6.1 Sol (`gpt-6.1-sol`, esfuerzo alto) | Codex CLI | `prompt_llm_codex.md` | `correr_lotes_codex.py` |
| `argon` | Gemini 4 Argon (`gemini-4-argon-high`) | Antigravity CLI (`agy`) | `prompt_llm_codex.md` | `correr_lotes_agy.py` |
| `fable` | Claude Fable 5.2 (`claude-fable-5-2`, esfuerzo alto) | Claude Code CLI (`claude -p`) | `prompt_llm_codex.md` | `correr_lotes_claude.py` |

Los dos prompts piden lo mismo. El de `o2` tiene las rutas del proyecto original; el de los scripts usa la carpeta aislada.

**Salida:** un JSON por corrida y lote, `data/llm/resultados/<corrida>_<lote>.json`, con la forma
`{"10001": {"p": 0.70, "duda": ""}, ...}`. `p` es la probabilidad de usado (entre 0.02 y 0.98) y `duda` es una pregunta general
cuando el LLM no está seguro. Antes de guardar, los scripts validan que estén todos los `id` y que cada `p` esté entre 0 y 1.

**Costo de referencia:** unos 100.000 tokens por lote de 100 filas.

## Cómo replicarlo

### Con los scripts (mismo proceso)

Desde la raíz del repositorio, con la CLI correspondiente instalada y logueada:

```bash
python3 llm/correr_lotes_codex.py --version sol --lotes A --prueba 5     # prueba de humo: 5 filas, no guarda
python3 llm/correr_lotes_codex.py --version sol --lotes A B C V1 V2 V3 V4
python3 llm/correr_lotes_agy.py --version argon --lotes T01 T02
python3 llm/correr_lotes_claude.py --version fable --lotes R01 R02
```

Las rutas de las CLIs (`CODEX`, `AGY`) están fijadas a la instalación de macOS donde se corrieron; cambiarlas si hace falta.
Para usar otro modelo: `--modelo <nombre>`.

### Con cualquier LLM (aproximado)

Para cualquier LLM de chat o API: adjuntar `guia_llm_dirigido_v2.md` y un lote de `data/llm/lotes/`, y pedir el resultado con
este prompt (en inglés, como las corridas originales). Reemplazar `<N>`, `<FIRST_ID>` y `<LAST_ID>` con los valores del lote.

```text
You are classifying MercadoLibre Argentina listings from 2015 as new or used.

You have two attachments: the guideline (guia_llm_dirigido_v2.md) and a CSV with <N> listings (ids <FIRST_ID>–<LAST_ID>).
Read the guideline first and follow it exactly. Use nothing else: no web search, no other files.

For EVERY row, read it yourself and decide. Do not write code that scores rows. Give p = probability that the item is USED,
between 0.02 and 0.98. Use values near 0.5 when the text gives you nothing specific about the product.

If you are genuinely unsure (p between 0.35 and 0.65), or your answer rests on an assumption you are not confident about, add a
short doubt phrased as a GENERAL question a human expert could answer once for many listings. Otherwise leave it empty.

Answer only with JSON, all <N> ids, in this format:
{"<FIRST_ID>": {"p": 0.70, "duda": ""}, ...}
```

Guardar la respuesta como `data/llm/resultados/<corrida>_<lote>.json`. Para que el modelo use la nueva corrida en lugar de una de
las originales, el nombre de la corrida tiene que coincidir con el que espera `src/fusiones.py` (`o2`, `sol`, `argon` o `fable`).
Después hay que volver a correr el notebook 1, porque las lecturas de train forman parte del contexto de las fusiones, y el
notebook 2.

**Qué esperar:** correlación alta con las lecturas originales, pero no igualdad. Algunas etiquetas finales de las zonas pueden
cambiar.
