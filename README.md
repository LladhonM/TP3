# ¿Nuevo o usado? — TP3 Análisis Predictivo (ITBA)

Clasificador de publicaciones de MercadoLibre (`new` / `used`) para la competencia de Kaggle del TP3. Métrica: accuracy.

| | |
|---|---|
| Submission enviada | `submission/submission_final.csv` |
| Kaggle público | **0.95552** |
| CV estimada | 0.95389 |

La CV estimada es la del modelo con la banda en TabPFN v3.5 (0.95381) más lo que sumó TabPFN-3.5-Thinking en la CV de la
banda (+5.8 filas sobre 70.000).

## Contenido

```
presentacion/                  presentación del examen (PDF)
notebooks/
  1_entrenar_modelo_final.ipynb  entrena el modelo final desde los datos y lo guarda en modelo/
  2_predecir_test.ipynb          aplica el modelo guardado al test y escribe submission/submission.csv
src/                           funciones que usan los notebooks
  features.py, encoders_oof.py, features_perfil.py, features_conocimiento.py, cv_anidada.py, modelado.py
                                 features y CV anidada (target encodings y texto dentro de cada fold)
  modelo_final.py                capa 1: LightGBM, CatBoost, TabPFN y meta-modelo
  features_texto_grupo.py        3 features de texto por grupo para las fusiones
  fusiones.py                    capa 2: fusiones con lecturas de LLM en tres zonas (TabPFN v3.5 en anillo y bolsillo)
  banda_thinking.py              capa 2, banda: TabPFN-3.5-Thinking por la API de Prior Labs
data/llm/                      lecturas de los LLM (insumo fijo del modelo)
  clave*.csv                     qué fila de train/test es cada lectura y a qué zona pertenece
  resultados/                    P(usado) que dio cada LLM a cada fila, por lote
  lotes/                         las filas tal como se les mostraron a los LLM
data/thinking/test_banda.csv   P(usado) de TabPFN-3.5-Thinking en la banda de test (insumo fijo del modelo)
llm/                           para volver a correr las lecturas: guía congelada, prompts y scripts de las corridas
requirements_thinking.txt      entorno aparte para volver a correr la banda por la API (opcional)
submission/submission_final.csv  la submission enviada (el notebook 2 compara contra ella)
```

## El modelo final

**Capa 1 (todas las filas):** regresión logística sobre los logits de tres modelos, entrenados con CV estratificada de 5 folds
(semilla 42) y reentrenados con el 100% de train.
- **LightGBM:** features estructuradas del JSON, target encodings y conteos de vendedor/categoría/ciudad, predicciones OOF de
  modelos de texto (TF-IDF + regresión logística), perfiles de vendedor y categoría y 9 reglas de "conocimiento del mundo".
  Hiperparámetros de Optuna, 3 semillas.
- **CatBoost:** las features base con `seller_id` crudo como categórica.
- **TabPFN v3.5:** las features del LightGBM (153 columnas), en contexto.

Todo lo que usa la etiqueta (target encodings, modelos de texto) se ajusta **dentro de cada fold** (CV anidada).

**Capa 2 (3.052 filas de test):** en tres zonas donde la capa 1 duda, una fusión TabPFN combina la probabilidad de la capa 1, la
P(usado) que dieron LLMs leyendo la publicación con una guía escrita y congelada (Claude Opus, GPT Sol, Gemini Argon, Claude
Fable) y las features de la fila.

En la banda, la fusión es **TabPFN-3.5-Thinking** (`thinking_effort='mini'`, `thinking_metric='accuracy'`), que corre en los
servidores de Prior Labs a través de su API (`tabpfn_client`). Recibe las mismas entradas que la fusión TabPFN v3.5 a la que
reemplazó: en la CV de la banda (10 folds × 5 semillas, corte 0.5) sumó +278.8 aciertos netos (filas que corrige menos filas que
rompe) contra +273.0 de la v3.5, es decir +5.8, y fue mejor en 4 de las 5 semillas. En anillo y bolsillo la fusión sigue siendo
TabPFN v3.5 local.

| Zona | Filas de test | Fusión | LLM | Contexto de train | Corte |
|---|---|---|---|---|---|
| Banda | 1.476 | TabPFN-3.5-Thinking 'mini' (API de Prior Labs) | Opus, Sol, Argon | 3.084 | 0.5 |
| Anillo | 1.027 | TabPFN v3.5 | Opus, Fable, Sol, Argon | 2.575 | 0.5 |
| Bolsillo | 549 | TabPFN v3.5 | Opus, Sol | 3.884 (banda + 800) | 0.39 |

Correr la fusión de la banda sube sus filas (las 3.084 de contexto con su etiqueta y las 1.476 de test, cada una con la capa 1,
las lecturas de los LLM y las 153 features) a los servidores de Prior Labs. Por eso el modelo usa su predicción guardada y la API
solo se llama si se pide de forma explícita (ver "Cómo reproducir", paso 6).

## Cómo reproducir

1. **Datos:** bajar `train_data.jsonlines` y `test_data.jsonlines` de la competencia y ponerlos en `data/`.
2. **Entorno:** `pip install -r requirements.txt`. TabPFN v3.5 pide aceptar su licencia: definir la variable de entorno
   `TABPFN_TOKEN` con un token de Prior Labs.
3. **GPU:** TabPFN predice con 56.000–70.000 filas de contexto y necesita una GPU con CUDA. Se corrió en Kaggle con una T4:
   subir esta carpeta como dataset, copiarla a `/kaggle/working`, activar *GPU T4* e *Internet* y agregar `TABPFN_TOKEN` en
   *Secrets*.
4. Correr `notebooks/1_entrenar_modelo_final.ipynb` (~2 h): deja el modelo en `modelo/`.
5. Correr `notebooks/2_predecir_test.ipynb` (~40 min): escribe `submission/submission.csv` y la compara con la enviada. En la
   banda usa la predicción guardada de Thinking (`data/thinking/test_banda.csv`) y no llama a la API; además deja las matrices
   de la banda en `modelo/banda_thinking.npz`.
6. **Opcional, volver a correr la banda Thinking por la API** (consume créditos de la cuenta de Prior Labs y sube las filas de la
   banda a sus servidores):
   - En un entorno aparte: `pip install -r requirements_thinking.txt`. `tabpfn-client` 0.7.1 pide `pandas<=2.3.3` y no se puede
     instalar junto con `requirements.txt`.
   - Token: `tabpfn_client` toma el token de la API de la variable de entorno `TABPFN_TOKEN` (o de
     `tabpfn_client.set_access_token(...)` dentro de la sesión). Se genera en la cuenta de Prior Labs (platform.priorlabs.ai) y
     no se guarda en el repositorio.
   - Desde la raíz: `python src/banda_thinking.py --correr-api`. Lee `modelo/banda_thinking.npz`, hace un ajuste de
     TabPFN-3.5-Thinking (el de la submission tardó unos 2,5 minutos) y escribe `modelo/banda_thinking_api.csv`. Sin
     `--correr-api` solo muestra qué se subiría.
   - En el notebook 2, poner `USAR_BANDA_THINKING_NUEVA = True` y volver a correr desde la sección 2.

## Qué se replica exactamente y qué no

- **Features, CatBoost, LightGBM y meta-modelo:** corrida desde los datos crudos, la versión de este repositorio reproduce
  exactamente los resultados que generaron la submission (ver "Verificación").
- **TabPFN v3.5** es determinístico en el mismo tipo de hardware, pero entre GPUs distintas puede variar en el tercer decimal de
  la probabilidad. Medido: la misma fusión en CUDA y en la GPU de una Mac difiere como máximo en 0.0026 y cambia 2 de 1.476
  etiquetas. La capa 1 de TabPFN corrió en Colab (T4) y las fusiones de anillo y bolsillo en Kaggle (T4).
- **Banda (TabPFN-3.5-Thinking):** su predicción de test es un insumo fijo (`data/thinking/test_banda.csv`) porque el modelo
  corre en los servidores de Prior Labs, no en este código. Con `src/banda_thinking.py` se puede volver a pedir, pero cada
  corrida consume créditos y no devuelve exactamente las mismas probabilidades. Medido el 2026-11-13: el mismo ajuste,
  repetido una vez con las mismas matrices y parámetros (`random_state` 0, el valor por defecto del cliente), difiere de la
  corrida de la submission como máximo en 0.013 de probabilidad (mediana 0.002; correlación de logits 0.9999) y cambia 6 de
  las 1.476 etiquetas de la banda, es decir 6 de las 30.000 filas de la submission. Las 6 tienen probabilidad entre 0.495 y
  0.506 en las dos corridas, pegadas al corte de 0.5. Además, la API usa la versión vigente del modelo de Prior Labs, que con
  Thinking no se puede fijar: si Prior Labs la actualiza, las diferencias pueden ser mayores.
- **Lecturas de los LLM:** son un insumo fijo (`data/llm/resultados`). Con `llm/` se pueden volver a correr, pero los LLM no son
  deterministas: dan lecturas parecidas, no idénticas (dos corridas de Opus con la misma guía correlacionan 0.96).
- **Zonas:** son las filas que leyeron los LLM (`data/llm/clave*.csv`). Se eligieron una vez, con la probabilidad del stacking
  de ese momento, y quedaron fijas como datos de entrada.

## Verificación

Antes de publicar el modelo con la banda en TabPFN v3.5, el código de `src/` se corrió desde los datos crudos (macOS, Apple M3,
Python 3.13, CPU) y cada resultado se comparó con el que generó aquella submission:

| Etapa | Resultado |
|---|---|
| Datasets aplanados y textos (train y test) | idénticos |
| Folds de la CV | idénticos |
| Features de texto por grupo | idénticas |
| Matrices de TabPFN (5 folds y 100%) e índices de categóricas | idénticas |
| LightGBM: OOF y test (con el modelo guardado y recargado) | idénticos |
| CatBoost: OOF y test (con el modelo guardado y recargado) | idénticos |
| Meta-modelo: OOF, test y umbral del resto (0.48, accuracy OOF 0.94701) | idénticos |
| Contextos y matrices de test de las tres fusiones | idénticos |
| Submission armada con esas piezas | 0 filas distintas de aquella submission |

Al pasar la banda a TabPFN-3.5-Thinking se verificó, también en CPU y sin llamar a la API:

| Etapa | Resultado |
|---|---|
| Matrices de la banda armadas con `src/fusiones.py` desde la capa 1 y las matrices de TabPFN de arriba (contexto 3.084 × 157 con su etiqueta, test 1.476 × 157, índices de categóricas, ids y filas) | idénticas a las que recibió la API |
| `guardar_matrices_banda` (lo que lee `src/banda_thinking.py`) | las mismas matrices |
| Filas de `data/thinking/test_banda.csv` | las 1.476 filas de la banda de `data/llm/clave_test.csv`, en el mismo orden |
| Submission armada como el notebook 2: capa 1, anillo y bolsillo como antes y la banda con el insumo fijo | 0 filas distintas de `submission_final.csv` |
| La misma, con la banda TabPFN v3.5 anterior (control) | 0 filas distintas de la submission anterior |
| Diferencia con la submission anterior | 62 etiquetas (34 pasan a `new`, 28 a `used`), todas en la banda |

Las predicciones de TabPFN (capa 1 y fusiones) no se pudieron recalcular en esa máquina porque necesitan CUDA; en la
verificación se usaron sus salidas originales (en la banda, el insumo fijo de Thinking). La pasada de validación de las
fusiones en Kaggle (T4), con el mismo código, reprodujo las tres zonas del modelo con la banda en TabPFN v3.5 con 0 etiquetas
distintas.
