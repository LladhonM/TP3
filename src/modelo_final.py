"""
Capa 1 del modelo final: LightGBM, CatBoost y TabPFN combinados con una regresión logística sobre sus logits.

Cada modelo tiene dos pasos: `entrenar_*` (CV anidada para el OOF que usa el meta-modelo + modelo reentrenado con el 100% de
train) y `predecir_*` (aplica el modelo guardado a la matriz de test).
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from cv_anidada import FACTOR_ITERACIONES_COMPLETO, entrenar_sobre_folds, preparar_completo, preparar_folds_anidados
from features_conocimiento import features_conocimiento
from features_perfil import agregar_features_perfil
from modelado import (N_HILOS, PARAMETROS_CATBOOST, alinear_categorias, entrenar_catboost_cv, entrenar_lgb_cv, mejor_umbral,
                      preparar_para_catboost, preparar_para_lgb)

# LightGBM: hiperparámetros elegidos con Optuna (la prueba de mayor accuracy entre las 10 de menor log-loss), 3 semillas
SEMILLAS_LGB = [42, 7, 2024]
PARAMETROS_LGB = {
    'objective': 'binary',
    'verbose': -1,
    'num_threads': N_HILOS,
    'bagging_freq': 1,
    'bagging_fraction': 0.8408083652566225,
    'cat_smooth': 19.847254293518994,
    'feature_fraction': 0.379656703629579,
    'lambda_l1': 0.001260834005751,
    'lambda_l2': 0.7303128553886228,
    'learning_rate': 0.023364310638159,
    'min_child_samples': 73,
    'min_split_gain': 0.2031706284247052,
    'num_leaves': 73,
}

# TabPFN: identificadores de alta cardinalidad afuera; predicción de a bloques de 14.000 filas, como en la corrida original
VERSION_TABPFN = 'v3.5'
COLUMNAS_FUERA_DE_TABPFN = ['id', 'seller_id', 'category_id', 'city_id']
BLOQUE_TABPFN = 14000

# Recorte del logit en el meta-modelo: evita infinitos con probabilidades 0 o 1
EPSILON_LOGIT = 1e-6


# ---------- Matrices de features ----------

def agregar_columnas_a_folds(folds_preparados: list[dict], folds: list, columnas_train: pd.DataFrame,
                             columnas_test: pd.DataFrame) -> list[dict]:
    """Suma columnas sin target a las matrices de cada fold. X_combinado tiene las filas [ajuste; validación] en ese orden."""
    nuevos = []
    for fold, (indices_ajuste, indices_validacion) in zip(folds_preparados, folds):
        filas = np.r_[indices_ajuste, indices_validacion]
        nuevos.append({
            **fold,
            'X_combinado': pd.concat([fold['X_combinado'], columnas_train.iloc[filas].reset_index(drop=True)], axis=1),
            'X_test_fold': pd.concat([fold['X_test_fold'], columnas_test.reset_index(drop=True)], axis=1),
        })
    return nuevos


def matrices_base(X_train, X_test, T_train, T_test, target, folds) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Features base con target encodings y modelos de texto ajustados dentro de cada fold (las del CatBoost)."""
    folds_preparados = preparar_folds_anidados(X_train, X_test, T_train, T_test, target, folds, nombre_cache='base')
    X_completo, X_test_completo = preparar_completo(X_train, X_test, T_train, T_test, target, nombre_cache='base')
    return folds_preparados, X_completo, X_test_completo


def matrices_perfil_conocimiento(X_train, X_test, T_train, T_test, target, folds) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Features base + perfiles de vendedor y categoría + 9 reglas de conocimiento del mundo (las del LightGBM y TabPFN)."""
    X_train, X_test = agregar_features_perfil(X_train, X_test)
    conocimiento_train = features_conocimiento(T_train)
    conocimiento_test = features_conocimiento(T_test)
    folds_preparados = preparar_folds_anidados(X_train, X_test, T_train, T_test, target, folds, nombre_cache='perfil')
    folds_preparados = agregar_columnas_a_folds(folds_preparados, folds, conocimiento_train, conocimiento_test)
    X_completo, X_test_completo = preparar_completo(X_train, X_test, T_train, T_test, target, nombre_cache='perfil')
    X_completo = pd.concat([X_completo, conocimiento_train], axis=1)
    X_test_completo = pd.concat([X_test_completo, conocimiento_test], axis=1)
    return folds_preparados, X_completo, X_test_completo


# ---------- LightGBM ----------

def entrenar_lightgbm(folds_preparados: list[dict], X_completo: pd.DataFrame, X_test_completo: pd.DataFrame,
                      target: pd.Series) -> tuple[np.ndarray, list[lgb.Booster], pd.DataFrame]:
    """OOF promedio de las 3 semillas y un modelo por semilla con el 100% de train (árboles = 1.2 × promedio de la CV)."""
    X_ajuste, X_test = alinear_categorias(preparar_para_lgb(X_completo), preparar_para_lgb(X_test_completo))
    oofs, modelos = [], []
    for semilla in SEMILLAS_LGB:
        parametros = {**PARAMETROS_LGB, 'seed': semilla}
        oof, _, iteraciones = entrenar_sobre_folds(folds_preparados, len(target), entrenar_lgb_cv, verbose=False, parametros=parametros)
        n_arboles = int(np.mean(iteraciones) * FACTOR_ITERACIONES_COMPLETO)
        modelos.append(lgb.train(parametros, lgb.Dataset(X_ajuste, target), num_boost_round=n_arboles))
        oofs.append(oof)
    return np.mean(oofs, axis=0), modelos, X_test


def predecir_lightgbm(modelos: list[lgb.Booster], X_test: pd.DataFrame) -> np.ndarray:
    return np.mean([modelo.predict(X_test) for modelo in modelos], axis=0)


# ---------- CatBoost ----------

def entrenar_catboost(folds_preparados: list[dict], X_completo: pd.DataFrame, X_test_completo: pd.DataFrame,
                      target: pd.Series) -> tuple[np.ndarray, CatBoostClassifier, pd.DataFrame]:
    """OOF de la CV anidada y modelo con el 100% de train; seller_id va crudo como categórica (ordered target statistics)."""
    oof, _, iteraciones = entrenar_sobre_folds(folds_preparados, len(target), entrenar_catboost_cv, verbose=False)
    n_arboles = int(np.mean(iteraciones) * FACTOR_ITERACIONES_COMPLETO)
    X_ajuste, categoricas = preparar_para_catboost(X_completo)
    X_test, _ = preparar_para_catboost(X_test_completo)
    modelo = CatBoostClassifier(**{**PARAMETROS_CATBOOST, 'iterations': n_arboles})
    modelo.fit(Pool(X_ajuste, target.reset_index(drop=True), cat_features=categoricas))
    return oof, modelo, X_test


def predecir_catboost(modelo: CatBoostClassifier, X_test: pd.DataFrame) -> np.ndarray:
    return modelo.predict_proba(Pool(X_test, cat_features=modelo.get_cat_feature_indices()))[:, 1]


# ---------- TabPFN ----------

def a_numerico(X: pd.DataFrame, categorias: dict[str, list]) -> np.ndarray:
    """Columnas categóricas a códigos enteros con categorías fijas (NaN = categoría desconocida)."""
    X = X.copy()
    for columna, valores in categorias.items():
        X[columna] = pd.Categorical(X[columna], categories=valores).codes.astype('float32')
        X.loc[X[columna] < 0, columna] = np.nan
    return X.astype('float32').values


def matrices_tabpfn(folds_preparados: list[dict], X_completo: pd.DataFrame, X_test_completo: pd.DataFrame,
                    target: pd.Series) -> tuple[list[dict], dict, list[int], list[str]]:
    """Matrices numéricas (153 columnas) por fold y al 100%, índices de las categóricas y nombres de las columnas."""
    columnas = list(X_completo.columns)
    no_numericas = [columna for columna in columnas if not pd.api.types.is_numeric_dtype(X_completo[columna])]
    categorias = {}
    for columna in no_numericas:
        valores = pd.concat([X_completo[columna], X_test_completo[columna]]).dropna().unique().tolist()
        categorias[columna] = sorted(valores)
    conservar = [i for i, columna in enumerate(columnas) if columna not in COLUMNAS_FUERA_DE_TABPFN]
    indices_categoricos = [conservar.index(columnas.index(columna)) for columna in no_numericas if columnas.index(columna) in conservar]

    folds_numericos = []
    for fold in folds_preparados:
        n_ajuste = len(fold['fold_unico'][0][0])
        X_combinado = a_numerico(fold['X_combinado'], categorias)[:, conservar]
        y_combinado = fold['y_combinado'].values.astype('int8')
        folds_numericos.append({
            'X_ajuste': X_combinado[:n_ajuste], 'y_ajuste': y_combinado[:n_ajuste],
            'X_validacion': X_combinado[n_ajuste:], 'indices_validacion': fold['indices_validacion'],
        })
    completo = {
        'X': a_numerico(X_completo, categorias)[:, conservar],
        'y': target.values.astype('int8'),
        'X_test': a_numerico(X_test_completo, categorias)[:, conservar],
    }
    nombres = [columnas[i] for i in conservar]
    return folds_numericos, completo, indices_categoricos, nombres


def ajustar_y_predecir_tabpfn(X_contexto: np.ndarray, y_contexto: np.ndarray, X_aplicar: np.ndarray,
                              indices_categoricos: list[int]) -> np.ndarray:
    """TabPFN aprende en contexto: el ajuste guarda las filas de train y la predicción las lee (requiere GPU con CUDA)."""
    modelo = TabPFNClassifier.create_default_for_version(
        ModelVersion(VERSION_TABPFN),
        device='auto',
        categorical_features_indices=indices_categoricos,
        ignore_pretraining_limits=True,
        fit_mode='fit_preprocessors',
    )
    modelo.fit(X_contexto, y_contexto)
    bloques = []
    for desde in range(0, len(X_aplicar), BLOQUE_TABPFN):
        bloques.append(modelo.predict_proba(X_aplicar[desde:desde + BLOQUE_TABPFN])[:, 1])
    # float32, como se guardaron las predicciones originales que entraron al meta-modelo
    return np.concatenate(bloques).astype('float32')


def oof_tabpfn(folds_numericos: list[dict], indices_categoricos: list[int], n_train: int) -> np.ndarray:
    # Contenedor float64 con valores float32: así entró el OOF original al meta-modelo (el test queda en float32)
    oof = np.full(n_train, np.nan)
    for numero, fold in enumerate(folds_numericos):
        inicio = time.time()
        prediccion = ajustar_y_predecir_tabpfn(fold['X_ajuste'], fold['y_ajuste'], fold['X_validacion'], indices_categoricos)
        oof[fold['indices_validacion']] = prediccion
        print(f'TabPFN fold {numero}: {(time.time() - inicio) / 60:.1f} min', flush=True)
    return oof


def features_oof_por_fila(folds_numericos: list[dict], n_train: int) -> np.ndarray:
    """Matriz de train con las filas de validación de cada fold: los encodings de cada fila se calcularon sin su etiqueta."""
    X = np.full((n_train, folds_numericos[0]['X_validacion'].shape[1]), np.nan)
    for fold in folds_numericos:
        X[fold['indices_validacion']] = fold['X_validacion']
    return X


# ---------- Meta-modelo ----------

def logit(probabilidades: np.ndarray) -> np.ndarray:
    recortadas = np.clip(probabilidades, EPSILON_LOGIT, 1 - EPSILON_LOGIT)
    return np.log(recortadas / (1 - recortadas))


def matriz_logits(predicciones: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([logit(prediccion) for prediccion in predicciones])


def oof_meta(oofs: list[np.ndarray], target: pd.Series, folds: list) -> tuple[np.ndarray, float, float]:
    """OOF del meta-modelo con los mismos folds (pesos aprendidos sin el fold evaluado), umbral óptimo y accuracy."""
    matriz = matriz_logits(oofs)
    y = target.values
    oof = np.zeros(len(y))
    for indices_ajuste, indices_validacion in folds:
        meta = LogisticRegression(C=1.0).fit(matriz[indices_ajuste], y[indices_ajuste])
        oof[indices_validacion] = meta.predict_proba(matriz[indices_validacion])[:, 1]
    umbral, acierto = mejor_umbral(oof, target)
    return oof, umbral, acierto


def entrenar_meta(oofs: list[np.ndarray], target: pd.Series) -> LogisticRegression:
    return LogisticRegression(C=1.0).fit(matriz_logits(oofs), target.values)


def predecir_meta(meta: LogisticRegression, predicciones_test: list[np.ndarray]) -> np.ndarray:
    return meta.predict_proba(matriz_logits(predicciones_test))[:, 1]
