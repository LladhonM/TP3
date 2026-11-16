"""
CV anidada: las features supervisadas (target encoding y texto OOF) se recalculan dentro de cada fold externo.

En cada fold externo k, la parte de entrenamiento A (folds != k) cumple el rol de "train" y el fold k el de "test":
- A recibe encodings OOF con folds internos sobre A,
- el fold k y el test real reciben encodings ajustados con todo A.
Así cada fold reproduce exactamente la situación train → test y la CV no puede ver labels de validación.

Preparar las matrices de cada fold es lo caro (~70 s por fold); entrenar el modelo es lo barato. Por eso la preparación
se puede guardar en disco (`nombre_cache`) y reutilizar, por ejemplo para probar muchos hiperparámetros.
"""
import pickle
import time
from typing import Callable

import numpy as np
import pandas as pd

from encoders_oof import agregar_encodings_de_grupo, agregar_predicciones_texto
from features import RUTA_PROCESADOS
from modelado import generar_folds

# Al reentrenar con 100% de train hay 25% más datos que en cada fold → algo más de árboles que el promedio de los folds
FACTOR_ITERACIONES_COMPLETO = 1.2
RUTA_CACHE = RUTA_PROCESADOS / 'cache_anidado'


def features_supervisadas_para_fold(X_ajuste: pd.DataFrame, X_aplicar: pd.DataFrame, T_ajuste: pd.DataFrame, T_aplicar: pd.DataFrame,
                                    y_ajuste: pd.Series, usar_texto: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds_internos = generar_folds(y_ajuste)
    X_ajuste, X_aplicar = agregar_encodings_de_grupo(X_ajuste, X_aplicar, y_ajuste, folds_internos)
    if usar_texto:
        X_ajuste, X_aplicar = agregar_predicciones_texto(X_ajuste, X_aplicar, T_ajuste, T_aplicar, y_ajuste, folds_internos)
    return X_ajuste, X_aplicar


def preparar_fold_anidado(X_train: pd.DataFrame, X_test: pd.DataFrame, T_train: pd.DataFrame, T_test: pd.DataFrame,
                          target: pd.Series, indices_ajuste: np.ndarray, indices_validacion: np.ndarray, usar_texto: bool = True) -> dict:
    """Matrices de un fold externo listas para entrenar: [A; fold k] con un único fold (ajuste = A, validación = fold k) y el test."""
    X_ajuste = X_train.iloc[indices_ajuste].reset_index(drop=True)
    T_ajuste = T_train.iloc[indices_ajuste].reset_index(drop=True)
    y_ajuste = target.iloc[indices_ajuste].reset_index(drop=True)

    # Validación y test se procesan juntos: ambos son "datos nunca vistos" para este fold
    X_aplicar = pd.concat([X_train.iloc[indices_validacion], X_test], ignore_index=True)
    T_aplicar = pd.concat([T_train.iloc[indices_validacion], T_test], ignore_index=True)
    X_ajuste, X_aplicar = features_supervisadas_para_fold(X_ajuste, X_aplicar, T_ajuste, T_aplicar, y_ajuste, usar_texto)

    n_validacion = len(indices_validacion)
    X_validacion, X_test_fold = X_aplicar.iloc[:n_validacion], X_aplicar.iloc[n_validacion:].reset_index(drop=True)
    return {
        'X_combinado': pd.concat([X_ajuste, X_validacion], ignore_index=True),
        'y_combinado': pd.concat([y_ajuste, target.iloc[indices_validacion]], ignore_index=True),
        'fold_unico': [(np.arange(len(X_ajuste)), np.arange(len(X_ajuste), len(X_ajuste) + n_validacion))],
        'X_test_fold': X_test_fold,
        'indices_validacion': indices_validacion,
    }


def preparar_folds_anidados(X_train: pd.DataFrame, X_test: pd.DataFrame, T_train: pd.DataFrame, T_test: pd.DataFrame,
                            target: pd.Series, folds: list, usar_texto: bool = True, nombre_cache: str | None = None) -> list[dict]:
    ruta = RUTA_CACHE / f'{nombre_cache}_folds.pkl' if nombre_cache else None
    if ruta is not None and ruta.exists():
        with open(ruta, 'rb') as archivo:
            return pickle.load(archivo)
    preparados = [
        preparar_fold_anidado(X_train, X_test, T_train, T_test, target, ajuste, validacion, usar_texto)
        for ajuste, validacion in folds
    ]
    if ruta is not None:
        RUTA_CACHE.mkdir(parents=True, exist_ok=True)
        with open(ruta, 'wb') as archivo:
            pickle.dump(preparados, archivo)
    return preparados


def entrenar_sobre_folds(folds_preparados: list[dict], n_train: int, entrenar_modelo: Callable, verbose: bool = True,
                         **argumentos_modelo) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    entrenar_modelo(X_comb, X_test, y_comb, folds, iteraciones, ...) es una función de modelado.py.
    Devuelve (oof, predicción de test promediando los modelos de fold, mejores iteraciones por fold).
    """
    iteraciones = []
    oof = np.zeros(n_train)
    prediccion_test = np.zeros(len(folds_preparados[0]['X_test_fold']))
    for numero_fold, fold in enumerate(folds_preparados):
        inicio = time.time()
        oof_combinado, prediccion_test_fold, _ = entrenar_modelo(
            fold['X_combinado'], fold['X_test_fold'], fold['y_combinado'], fold['fold_unico'], iteraciones, **argumentos_modelo,
        )
        n_ajuste = len(fold['fold_unico'][0][0])
        oof[fold['indices_validacion']] = oof_combinado[n_ajuste:]
        prediccion_test += prediccion_test_fold / len(folds_preparados)
        if verbose:
            y_validacion = fold['y_combinado'].values[n_ajuste:]
            acierto = ((oof[fold['indices_validacion']] > 0.5).astype(int) == y_validacion).mean()
            print(f'fold {numero_fold}: acc@0.5 = {acierto:.4f} | iteraciones = {iteraciones[-1]} | {time.time() - inicio:.0f}s', flush=True)
    return oof, prediccion_test, iteraciones


def preparar_completo(X_train: pd.DataFrame, X_test: pd.DataFrame, T_train: pd.DataFrame, T_test: pd.DataFrame,
                      target: pd.Series, usar_texto: bool = True, nombre_cache: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encodings ajustados con todo train (OOF interno para las filas de train) para el reentrenamiento al 100%."""
    ruta = RUTA_CACHE / f'{nombre_cache}_completo.pkl' if nombre_cache else None
    if ruta is not None and ruta.exists():
        with open(ruta, 'rb') as archivo:
            return pickle.load(archivo)
    matrices = features_supervisadas_para_fold(
        X_train.reset_index(drop=True), X_test.reset_index(drop=True),
        T_train.reset_index(drop=True), T_test.reset_index(drop=True), target.reset_index(drop=True), usar_texto,
    )
    if ruta is not None:
        RUTA_CACHE.mkdir(parents=True, exist_ok=True)
        with open(ruta, 'wb') as archivo:
            pickle.dump(matrices, archivo)
    return matrices
