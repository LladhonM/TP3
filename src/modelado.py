"""Utilidades de entrenamiento compartidas por todos los notebooks de modelos."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from features import COLUMNAS_CATEGORICAS

SEMILLA = 42
N_FOLDS = 5
# Fijo en 6: el número de hilos forma parte de la configuración que se replica
N_HILOS = 6

# seller/category/city van como target encoding + conteo; crudos tienen demasiada cardinalidad
CATEGORICAS_ALTA_CARDINALIDAD = ['seller_id', 'category_id', 'city_id']
PARAMETROS_LGB = {
    'objective': 'binary',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'min_child_samples': 20,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l2': 1.0,
    'verbose': -1,
    'seed': SEMILLA,
    'num_threads': N_HILOS,
}


def generar_folds(target: pd.Series) -> list:
    validacion = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    return list(validacion.split(np.zeros(len(target)), target))


def preparar_para_lgb(X: pd.DataFrame) -> pd.DataFrame:
    X = X.drop(columns=['id'] + CATEGORICAS_ALTA_CARDINALIDAD)
    for columna in COLUMNAS_CATEGORICAS:
        if columna in X.columns:
            X[columna] = X[columna].fillna('NA').astype(str)
    return X


def alinear_categorias(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for columna in COLUMNAS_CATEGORICAS:
        if columna not in X_train.columns:
            continue
        categorias = sorted(set(X_train[columna]) | set(X_test[columna]))
        X_train[columna] = pd.Categorical(X_train[columna], categories=categorias)
        X_test[columna] = pd.Categorical(X_test[columna], categories=categorias)
    return X_train, X_test


def mejor_umbral(probabilidades: np.ndarray, target: pd.Series) -> tuple[float, float]:
    umbrales = np.arange(0.30, 0.70, 0.005)
    accuracies = [((probabilidades > umbral).astype(int) == target).mean() for umbral in umbrales]
    indice_mejor = int(np.argmax(accuracies))
    return umbrales[indice_mejor], accuracies[indice_mejor]


def entrenar_lgb_cv(X_train: pd.DataFrame, X_test: pd.DataFrame, target: pd.Series, folds: list,
                    iteraciones: list | None = None, parametros: dict | None = None) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Si se pasa la lista `iteraciones`, se le agrega la mejor iteración de cada fold (para reentrenar al 100%)."""
    X_train, X_test = alinear_categorias(preparar_para_lgb(X_train), preparar_para_lgb(X_test))
    oof = np.zeros(len(X_train))
    prediccion_test = np.zeros(len(X_test))
    importancias = pd.Series(0.0, index=X_train.columns)
    for indices_ajuste, indices_validacion in folds:
        datos_ajuste = lgb.Dataset(X_train.iloc[indices_ajuste], target.iloc[indices_ajuste])
        datos_validacion = lgb.Dataset(X_train.iloc[indices_validacion], target.iloc[indices_validacion])
        modelo = lgb.train(
            parametros or PARAMETROS_LGB, datos_ajuste, num_boost_round=5000, valid_sets=[datos_validacion],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[indices_validacion] = modelo.predict(X_train.iloc[indices_validacion])
        prediccion_test += modelo.predict(X_test) / len(folds)
        importancias += pd.Series(modelo.feature_importance('gain'), index=X_train.columns)
        if iteraciones is not None:
            iteraciones.append(modelo.best_iteration)
    return oof, prediccion_test, importancias.sort_values(ascending=False)


PARAMETROS_CATBOOST = {
    'iterations': 4000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'eval_metric': 'Logloss',
    'random_seed': SEMILLA,
    'thread_count': N_HILOS,
    'verbose': 0,
    # Evita que CatBoost cree la carpeta catboost_info/ con logs en el directorio de trabajo
    'allow_writing_files': False,
}


def preparar_para_catboost(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """CatBoost recibe las categóricas crudas (incluido seller_id) y las codifica con ordered target statistics."""
    X = X.drop(columns=['id'])
    categoricas = [columna for columna in COLUMNAS_CATEGORICAS if columna in X.columns]
    for columna in categoricas:
        X[columna] = X[columna].fillna('NA').astype(str)
    return X, categoricas


def entrenar_catboost_cv(X_train: pd.DataFrame, X_test: pd.DataFrame, target: pd.Series, folds: list,
                         iteraciones: list | None = None, parametros: dict = PARAMETROS_CATBOOST) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    from catboost import CatBoostClassifier, Pool

    X_train, categoricas = preparar_para_catboost(X_train)
    X_test, _ = preparar_para_catboost(X_test)
    pool_test = Pool(X_test, cat_features=categoricas)
    oof = np.zeros(len(X_train))
    prediccion_test = np.zeros(len(X_test))
    importancias = pd.Series(0.0, index=X_train.columns)
    for indices_ajuste, indices_validacion in folds:
        pool_ajuste = Pool(X_train.iloc[indices_ajuste], target.iloc[indices_ajuste], cat_features=categoricas)
        pool_validacion = Pool(X_train.iloc[indices_validacion], target.iloc[indices_validacion], cat_features=categoricas)
        modelo = CatBoostClassifier(**parametros, early_stopping_rounds=300)
        modelo.fit(pool_ajuste, eval_set=pool_validacion, use_best_model=True)
        oof[indices_validacion] = modelo.predict_proba(pool_validacion)[:, 1]
        prediccion_test += modelo.predict_proba(pool_test)[:, 1] / len(folds)
        importancias += pd.Series(modelo.get_feature_importance(), index=X_train.columns)
        if iteraciones is not None:
            iteraciones.append(modelo.get_best_iteration())
    return oof, prediccion_test, importancias.sort_values(ascending=False)
