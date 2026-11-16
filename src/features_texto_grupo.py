"""
Tres features de texto por grupo (escritas por un LLM y elegidas por su aporte en las fusiones de anillo y bolsillo).

No usan la etiqueta: miden qué proporción de las publicaciones de un grupo (mismo título o misma categoría, en train + test)
dicen en el texto que el producto es usado.
"""
import numpy as np
import pandas as pd

from features_conocimiento import normalizar

PATRON_DICE_USADO = r'\busad[oa]s?\b|\bsemi ?nuev[oa]s?\b|\bcomo nuev[oa]s?\b|\bpoco uso\b|\bimpecable\b|\bfunciona\b|\bdetalle'
# Peso del promedio global en la proporción de cada grupo: los grupos chicos no dan proporciones extremas
PESO_PREVIO_TITULO = 3.0
PESO_PREVIO_CATEGORIA = 20.0


def tabla_train_y_test(X_train: pd.DataFrame, X_test: pd.DataFrame, T_train: pd.DataFrame, T_test: pd.DataFrame) -> pd.DataFrame:
    """Train (filas 0–69.999) y test (70.000–99.999) con las columnas que usan las tres features."""
    tabla = pd.concat([X_train, X_test], ignore_index=True)[['category_id', 'foto_pct_camara']]
    textos = pd.concat([T_train, T_test], ignore_index=True)
    tabla['titulo_normalizado'] = textos['titulo'].map(normalizar)
    tabla['garantia_normalizado'] = textos['garantia'].replace('sin_dato', '').map(normalizar)
    return tabla


def dice_usado(tabla: pd.DataFrame) -> pd.Series:
    texto = tabla['titulo_normalizado'] + ' | ' + tabla['garantia_normalizado']
    return texto.str.contains(PATRON_DICE_USADO, regex=True).astype(float)


def titulo_clave(tabla: pd.DataFrame) -> pd.Series:
    sin_signos = tabla['titulo_normalizado'].str.replace(r'[^a-z0-9 ]', ' ', regex=True)
    return sin_signos.str.replace(r'\s+', ' ', regex=True).str.strip()


def proporcion_suavizada(indicador: pd.Series, grupo: pd.Series, peso_previo: float) -> pd.Series:
    """Media del grupo encogida hacia la media global."""
    media_global = indicador.mean()
    suma = indicador.groupby(grupo).transform('sum')
    cantidad = indicador.groupby(grupo).transform('size')
    return (suma + peso_previo * media_global) / (cantidad + peso_previo)


def titulo_grupo_pct_dice_usado(tabla: pd.DataFrame) -> pd.Series:
    """Entre las publicaciones con el mismo título (de cualquier vendedor), cuántas dicen usado."""
    return proporcion_suavizada(dice_usado(tabla), titulo_clave(tabla), PESO_PREVIO_TITULO)


def categoria_dice_usado_menos_camara(tabla: pd.DataFrame) -> pd.Series:
    """Proporción de la categoría que dice usado menos su proporción de fotos de celular, ambas estandarizadas."""
    dice = proporcion_suavizada(dice_usado(tabla), tabla['category_id'], PESO_PREVIO_CATEGORIA)
    camara = tabla.groupby('category_id')['foto_pct_camara'].transform('mean')
    dice_estandarizada = (dice - dice.mean()) / dice.std()
    camara_estandarizada = (camara - camara.mean()) / camara.std()
    return dice_estandarizada - camara_estandarizada


def categoria_pct_dice_usado(tabla: pd.DataFrame) -> pd.Series:
    """Entre las publicaciones de la misma categoría, cuántas dicen usado."""
    return proporcion_suavizada(dice_usado(tabla), tabla['category_id'], PESO_PREVIO_CATEGORIA)


def features_texto_grupo(X_train, X_test, T_train, T_test) -> tuple[np.ndarray, np.ndarray]:
    """Las tres features en el orden en que entran a las fusiones: (train 70.000 × 3, test 30.000 × 3)."""
    tabla = tabla_train_y_test(X_train, X_test, T_train, T_test)
    columnas = [titulo_grupo_pct_dice_usado(tabla), categoria_dice_usado_menos_camara(tabla), categoria_pct_dice_usado(tabla)]
    valores = np.column_stack([pd.to_numeric(columna).to_numpy(dtype=float) for columna in columnas])
    n_train = len(X_train)
    return valores[:n_train], valores[n_train:]
