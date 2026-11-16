"""
Fusión de la banda con TabPFN-3.5-Thinking (esfuerzo 'mini', métrica accuracy) por la API de Prior Labs (`tabpfn_client`).

Mismas entradas que arma `fusiones.py` para la banda: logits de la capa 1 y de las lecturas de o2, sol y argon más las 153
features de la fila; contexto de 3.084 filas de train; corte 0.5. Lo que cambia es el modelo, que corre en los servidores de
Prior Labs: cada ajuste sube las filas de contexto y de test de la banda, consume créditos de la cuenta y no garantiza la misma
probabilidad en dos corridas.

Por eso el notebook 2 usa por defecto la predicción que generó la submission (`data/thinking/test_banda.csv`, insumo fijo) y
deja las matrices de la banda en `modelo/banda_thinking.npz`. Para volver a correr la banda por la API (entorno aparte con
`requirements_thinking.txt` y el token en `TABPFN_TOKEN`), desde la raíz del repositorio:
    python src/banda_thinking.py                 # solo muestra qué se subiría, no llama a la API
    python src/banda_thinking.py --correr-api    # ajusta y escribe modelo/banda_thinking_api.csv
Después, en el notebook 2, `USAR_BANDA_THINKING_NUEVA = True`.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import RAIZ

CARPETA_MODELO = RAIZ / 'modelo'
BANDA_THINKING_FIJA = RAIZ / 'data' / 'thinking' / 'test_banda.csv'
MATRICES_BANDA = CARPETA_MODELO / 'banda_thinking.npz'
BANDA_THINKING_NUEVA = CARPETA_MODELO / 'banda_thinking_api.csv'
ESFUERZO_THINKING = 'mini'
METRICA_THINKING = 'accuracy'


def guardar_matrices_banda(contexto: dict, filas: pd.DataFrame, X_aplicar: np.ndarray, ruta: Path = MATRICES_BANDA):
    """Contexto y test de la banda tal como los recibe la API, con el id de lectura y la fila de test de cada predicción."""
    np.savez_compressed(ruta, X_ajuste=contexto['X'], y_ajuste=contexto['y'], X_aplicar=X_aplicar,
                        categoricas=np.array(contexto['categoricos']), id=filas['id'].values, fila=filas.fila.values)


def probabilidad_banda(filas: pd.DataFrame, ruta: Path = BANDA_THINKING_FIJA) -> np.ndarray:
    """P(usado) de Thinking para las filas de test de la banda, en el mismo orden que `filas`."""
    guardada = pd.read_csv(ruta)
    mismas_filas = np.array_equal(guardada.fila.values, filas.fila.values)
    mismos_ids = np.array_equal(guardada['id'].values, filas['id'].values)
    if not (mismas_filas and mismos_ids):
        raise ValueError(f'{ruta.name}: las filas no coinciden con las de la banda de data/llm/clave_test.csv')
    return guardada.p_combinada.values


def ajustar_y_predecir_thinking(X_ajuste: np.ndarray, y_ajuste: np.ndarray, X_aplicar: np.ndarray,
                                categoricos: list[int]) -> np.ndarray:
    """Un ajuste de TabPFN-3.5-Thinking en la API: sube contexto y test, consume créditos y tarda unos minutos."""
    # tabpfn_client va en un entorno aparte (pide pandas<=2.3.3): solo se importa si se llama a la API
    import tabpfn_client

    modelo = tabpfn_client.TabPFNClassifier(
        thinking_mode=True,
        thinking_effort=ESFUERZO_THINKING,
        thinking_metric=METRICA_THINKING,
        categorical_features_indices=categoricos,
    )
    modelo.fit(X_ajuste, y_ajuste)
    columna_usado = list(modelo.classes_).index(1)
    return modelo.predict_proba(X_aplicar)[:, columna_usado]


def correr_api(matrices: dict, ruta_salida: Path):
    categoricos = [int(indice) for indice in matrices['categoricas']]
    inicio = time.time()
    probabilidad = ajustar_y_predecir_thinking(matrices['X_ajuste'], matrices['y_ajuste'], matrices['X_aplicar'], categoricos)
    resultado = pd.DataFrame({'id': matrices['id'], 'fila': matrices['fila'], 'p_combinada': probabilidad})
    resultado.to_csv(ruta_salida, index=False)
    print(f'{ruta_salida}: {len(resultado)} filas | usados {(probabilidad > 0.5).mean():.3f} | {time.time() - inicio:.0f}s')


def main():
    argumentos = argparse.ArgumentParser(description='Vuelve a correr la banda con TabPFN-3.5-Thinking por la API de Prior Labs.')
    argumentos.add_argument('--correr-api', action='store_true',
                            help='llama a la API: sube las filas de la banda a Prior Labs y consume créditos')
    argumentos.add_argument('--matrices', type=Path, default=MATRICES_BANDA, help='matrices que deja el notebook 2')
    argumentos.add_argument('--salida', type=Path, default=BANDA_THINKING_NUEVA)
    opciones = argumentos.parse_args()

    matrices = dict(np.load(opciones.matrices))
    print(f'contexto {matrices["X_ajuste"].shape} | test {matrices["X_aplicar"].shape} | '
          f'{len(matrices["categoricas"])} categóricas')
    if not opciones.correr_api:
        print('No se llamó a la API: agregar --correr-api para subir estas filas a Prior Labs y ajustar.')
        return
    correr_api(matrices, opciones.salida)


if __name__ == '__main__':
    main()
