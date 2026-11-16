"""
Capa 2 del modelo final: en tres zonas donde la capa 1 duda, una fusión TabPFN combina la probabilidad de la capa 1, las
lecturas de los LLM y las 153 features de la fila.

Las zonas son las filas que leyeron los LLM (`data/llm/clave*.csv`): se eligieron una vez y quedan fijas como datos de entrada.
- banda:    lecturas de Opus (o2), GPT Sol y Gemini Argon; contexto de 3.084 filas de train; corte 0.5; la fusión es
            TabPFN-3.5-Thinking por la API de Prior Labs (`banda_thinking.py`)
- anillo:   las mismas más Claude Fable; contexto de 2.575 filas; corte 0.5; suma las 3 features de texto por grupo
- bolsillo: Opus y Sol; contexto banda + 800 filas; corte 0.39; suma las 3 features de texto por grupo
Anillo y bolsillo usan TabPFN v3.5 local (`ajustar_y_predecir`).
"""
import json

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from features import RAIZ

CARPETA_LLM = RAIZ / 'data' / 'llm'
VERSION_TABPFN = 'v3.5'
# Recorte del logit de las probabilidades de entrada (capa 1 y LLM)
EPSILON_LOGIT = 1e-4
# Sin CUDA (Mac): modo de bajo consumo y predicción de a 100 filas para que el contexto entre en memoria
OPCIONES_AHORRO = {'memory_saving_mode': True, 'fit_mode': 'low_memory', 'keep_cache_on_device': False}
BLOQUE_SIN_CUDA = 100

LOTES_BANDA_ORIGINALES = ['A', 'B', 'C', 'V1', 'V2', 'V3', 'V4']
LOTES_BANDA_EXTRA = [f'E{n:02d}' for n in range(1, 9)]
LOTES_BANDA_NUEVOS = [f'N{n:02d}' for n in range(1, 17)]
CONTEXTO_BANDA = [('clave.csv', LOTES_BANDA_ORIGINALES), ('clave_extra.csv', LOTES_BANDA_EXTRA), ('clave_nueva.csv', LOTES_BANDA_NUEVOS)]

ZONAS = {
    'banda': {
        # Thinking ganó a TabPFN v3.5 con las mismas entradas: CV 10 folds x 5 semillas, +5.8 filas netas (4 de 5 semillas)
        'modelo': 'tabpfn_thinking',
        'corridas': ['o2', 'sol', 'argon'],
        'usa_texto_grupo': False,
        'corte': 0.5,
        'contexto': CONTEXTO_BANDA,
        'test': ('clave_test.csv', [f'T{n:02d}' for n in range(1, 16)]),
    },
    'anillo': {
        'modelo': 'tabpfn',
        'corridas': ['o2', 'fable', 'sol', 'argon'],
        'usa_texto_grupo': True,
        'corte': 0.5,
        'contexto': [
            ('clave_anillo.csv', [f'R{n:02d}' for n in range(1, 9)]),
            ('clave_anillo_nueva.csv', [f'RN{n:02d}' for n in range(1, 18)]),
            ('clave_anillo_66.csv', ['RX01']),
        ],
        'test': ('clave_test_anillo.csv', [f'TR{n:02d}' for n in range(1, 12)]),
    },
    'bolsillo': {
        'modelo': 'tabpfn',
        'corridas': ['o2', 'sol'],
        'usa_texto_grupo': True,
        # La fusión del bolsillo se queda corta al llamar "usado" (75% de usados en la zona): corte elegido con CV anidada
        'corte': 0.39,
        'contexto': CONTEXTO_BANDA + [('clave_bolsillo.csv', [f'P{n:02d}' for n in range(1, 9)])],
        'test': ('clave_test_bolsillo.csv', [f'TP{n:02d}' for n in range(1, 7)]),
    },
}


def logit(probabilidades: np.ndarray) -> np.ndarray:
    recortadas = np.clip(probabilidades, EPSILON_LOGIT, 1 - EPSILON_LOGIT)
    return np.log(recortadas / (1 - recortadas))


def leer_corrida(corrida: str, lotes: list[str]) -> dict[int, float]:
    """P(usado) que dio un LLM a cada publicación de los lotes, por id de lectura."""
    respuestas = {}
    for lote in lotes:
        with open(CARPETA_LLM / 'resultados' / f'{corrida}_{lote}.json') as archivo:
            for id_lectura, respuesta in json.load(archivo).items():
                respuestas[int(id_lectura)] = float(respuesta['p'])
    return respuestas


def filas_leidas(archivo_clave: str, lotes: list[str], corridas: list[str], probabilidad_capa1: np.ndarray) -> pd.DataFrame:
    """Filas de los lotes con su número de fila, etiqueta (en train), lectura de cada LLM y probabilidad de la capa 1."""
    clave = pd.read_csv(CARPETA_LLM / archivo_clave)
    filas = clave[clave.lote.isin(lotes)].copy()
    for corrida in corridas:
        respuestas = leer_corrida(corrida, lotes)
        faltantes = set(filas.id) - set(respuestas)
        if faltantes:
            raise ValueError(f'faltan {len(faltantes)} lecturas de {corrida} en los lotes {lotes}')
        filas[corrida] = filas.id.map(respuestas)
    filas['p_capa1'] = probabilidad_capa1[filas.fila.values]
    return filas


def matriz_fusion(filas: pd.DataFrame, corridas: list[str], X_filas: np.ndarray, indices_categoricos: list[int],
                  texto_grupo: np.ndarray | None) -> tuple[np.ndarray, list[int]]:
    """Logits (capa 1 y LLM), las 153 features de cada fila y, si la zona las usa, las 3 de texto por grupo."""
    logits = np.column_stack([logit(filas.p_capa1.values)] + [logit(filas[corrida].values) for corrida in corridas])
    bloques = [logits, X_filas[filas.fila.values]]
    if texto_grupo is not None:
        bloques.append(texto_grupo[filas.fila.values])
    categoricos_en_matriz = [logits.shape[1] + indice for indice in indices_categoricos]
    return np.column_stack(bloques), categoricos_en_matriz


def contexto_de_zona(zona: str, probabilidad_capa1_oof: np.ndarray, X_train_oof: np.ndarray, indices_categoricos: list[int],
                     texto_grupo_train: np.ndarray) -> dict:
    """Lo que aprende la fusión de la zona: filas de train leídas, con el OOF de la capa 1 y features sin su etiqueta."""
    configuracion = ZONAS[zona]
    partes = [filas_leidas(clave, lotes, configuracion['corridas'], probabilidad_capa1_oof) for clave, lotes in configuracion['contexto']]
    filas = pd.concat(partes, ignore_index=True)
    texto_grupo = texto_grupo_train if configuracion['usa_texto_grupo'] else None
    X, categoricos = matriz_fusion(filas, configuracion['corridas'], X_train_oof, indices_categoricos, texto_grupo)
    return {'X': X, 'y': filas.y.values, 'categoricos': categoricos}


def filas_de_test(zona: str, probabilidad_capa1_test: np.ndarray, X_test: np.ndarray, indices_categoricos: list[int],
                  texto_grupo_test: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    configuracion = ZONAS[zona]
    clave, lotes = configuracion['test']
    filas = filas_leidas(clave, lotes, configuracion['corridas'], probabilidad_capa1_test)
    texto_grupo = texto_grupo_test if configuracion['usa_texto_grupo'] else None
    X, _ = matriz_fusion(filas, configuracion['corridas'], X_test, indices_categoricos, texto_grupo)
    return filas, X


def ajustar_y_predecir(contexto: dict, X_aplicar: np.ndarray) -> np.ndarray:
    """TabPFN v3.5 con el contexto de la zona; con CUDA predice todo junto, sin CUDA de a bloques y en modo de bajo consumo."""
    en_cuda = torch.cuda.is_available()
    opciones = {} if en_cuda else OPCIONES_AHORRO
    modelo = TabPFNClassifier.create_default_for_version(
        ModelVersion(VERSION_TABPFN),
        device='auto',
        categorical_features_indices=contexto['categoricos'],
        random_state=0,
        **opciones,
    )
    modelo.fit(contexto['X'], contexto['y'])
    bloque = len(X_aplicar) if en_cuda else BLOQUE_SIN_CUDA
    probabilidades = []
    for desde in range(0, len(X_aplicar), bloque):
        probabilidades.append(modelo.predict_proba(X_aplicar[desde:desde + bloque])[:, 1])
    return np.concatenate(probabilidades)
