"""Features que usan el target: siempre out-of-fold en train para no filtrar el label."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Procesos para entrenar en paralelo los modelos de texto de cada fold (+1 = el modelo con todo train)
N_PROCESOS_TEXTO = 6

# Peso del prior en el target encoding: un grupo con 20 filas pesa lo mismo que el promedio global
SUAVIZADO_TARGET_ENCODING = 20


def media_suavizada(claves_ajuste: pd.Series, target: pd.Series, claves_aplicar: pd.Series, prior: float) -> np.ndarray:
    estadisticas = target.groupby(claves_ajuste.values).agg(['sum', 'count'])
    suavizada = (estadisticas['sum'] + prior * SUAVIZADO_TARGET_ENCODING) / (estadisticas['count'] + SUAVIZADO_TARGET_ENCODING)
    return claves_aplicar.map(suavizada).fillna(prior).values


def target_encoding_oof(columna_train: pd.Series, columna_test: pd.Series, target: pd.Series, folds: list) -> tuple[np.ndarray, np.ndarray]:
    prior = target.mean()
    oof = np.zeros(len(columna_train))
    for indices_ajuste, indices_validacion in folds:
        oof[indices_validacion] = media_suavizada(
            columna_train.iloc[indices_ajuste], target.iloc[indices_ajuste], columna_train.iloc[indices_validacion], prior,
        )
    prediccion_test = media_suavizada(columna_train, target, columna_test, prior)
    return oof, prediccion_test


def conteo_en_train_y_test(columna_train: pd.Series, columna_test: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    conteos = pd.concat([columna_train, columna_test]).value_counts()
    return columna_train.map(conteos).values, columna_test.map(conteos).values


def columnas_de_grupo(X: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        'seller_id': X['seller_id'],
        'category_id': X['category_id'],
        'city_id': X['city_id'],
        'seller_x_listing': X['seller_id'] + '|' + X['listing_type_id'],
        'seller_x_category': X['seller_id'] + '|' + X['category_id'],
        'category_x_listing': X['category_id'] + '|' + X['listing_type_id'],
        'primera_foto_max_size': X['primera_foto_max_size'],
    }


def agregar_encodings_de_grupo(X_train: pd.DataFrame, X_test: pd.DataFrame, target: pd.Series, folds: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, X_test = X_train.copy(), X_test.copy()
    grupos_train = columnas_de_grupo(X_train)
    grupos_test = columnas_de_grupo(X_test)
    for nombre, columna_train in grupos_train.items():
        columna_test = grupos_test[nombre]
        X_train[f'te_{nombre}'], X_test[f'te_{nombre}'] = target_encoding_oof(columna_train, columna_test, target, folds)
        X_train[f'cnt_{nombre}'], X_test[f'cnt_{nombre}'] = conteo_en_train_y_test(columna_train, columna_test)
    return X_train, X_test


def construir_vectorizadores() -> dict:
    return {
        'titulo_palabras': ('titulo', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents='unicode')),
        'titulo_chars': ('titulo', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, strip_accents='unicode')),
        'garantia': ('garantia', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents='unicode')),
        'atributos': ('atributos', TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents='unicode')),
    }


def ajustar_logistica_y_predecir(matriz_ajuste, y_ajuste: np.ndarray, matriz_aplicar, regularizacion: float) -> np.ndarray:
    modelo = LogisticRegression(C=regularizacion, max_iter=2000, solver='liblinear')
    modelo.fit(matriz_ajuste, y_ajuste)
    return modelo.predict_proba(matriz_aplicar)[:, 1]


def predecir_texto_oof(matriz_train, matriz_test, target: pd.Series, folds: list, regularizacion: float) -> tuple[np.ndarray, np.ndarray]:
    """Los 5 modelos de fold y el de 100% train son independientes: se entrenan en paralelo."""
    # liblinear reordena índices in-place y en los workers de joblib la matriz es de solo lectura
    matriz_train.sort_indices()
    matriz_test.sort_indices()
    y = target.values
    tareas = [(matriz_train[ajuste], y[ajuste], matriz_train[validacion]) for ajuste, validacion in folds]
    tareas.append((matriz_train, y, matriz_test))
    predicciones = Parallel(n_jobs=N_PROCESOS_TEXTO, max_nbytes=None)(
        delayed(ajustar_logistica_y_predecir)(matriz_ajuste, y_ajuste, matriz_aplicar, regularizacion)
        for matriz_ajuste, y_ajuste, matriz_aplicar in tareas
    )
    oof = np.zeros(matriz_train.shape[0])
    for (_, validacion), prediccion in zip(folds, predicciones[:-1]):
        oof[validacion] = prediccion
    return oof, predicciones[-1]


def agregar_predicciones_texto(X_train: pd.DataFrame, X_test: pd.DataFrame, T_train: pd.DataFrame, T_test: pd.DataFrame,
                               target: pd.Series, folds: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, X_test = X_train.copy(), X_test.copy()
    matrices_train, matrices_test = [], []
    for nombre, (columna, vectorizador) in construir_vectorizadores().items():
        # El vocabulario se ajusta sobre train+test: no usa el target, así que no filtra el label
        vectorizador.fit(pd.concat([T_train[columna], T_test[columna]]))
        matriz_train = vectorizador.transform(T_train[columna])
        matriz_test = vectorizador.transform(T_test[columna])
        X_train[f'txt_{nombre}'], X_test[f'txt_{nombre}'] = predecir_texto_oof(matriz_train, matriz_test, target, folds, regularizacion=4.0)
        matrices_train.append(matriz_train)
        matrices_test.append(matriz_test)

    matriz_completa_train = hstack(matrices_train).tocsr()
    matriz_completa_test = hstack(matrices_test).tocsr()
    X_train['txt_todo'], X_test['txt_todo'] = predecir_texto_oof(matriz_completa_train, matriz_completa_test, target, folds, regularizacion=4.0)
    return X_train, X_test
