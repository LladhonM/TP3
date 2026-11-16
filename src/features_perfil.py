"""
Perfil de vendedor y de categoría, calculado sobre train + test SIN usar el target (no puede filtrar el label).

Dos tipos de features:
- Perfil del grupo: cómo es el vendedor en general (¿publica con stock? ¿usa listados gratis? ¿pone garantía?).
- Desvío de la fila respecto de su grupo: dentro de un mismo vendedor o categoría, el usado suele ser "el raro"
  (más barato que la mediana, una sola unidad, otro tipo de publicación).
"""
import numpy as np
import pandas as pd

VARIABLES_PERFIL_VENDEDOR = {
    'log_price': ['mean', 'median', 'std'],
    'log_initial_quantity': ['mean'],
    'es_unidad_unica': ['mean'],
    'log_sold_quantity': ['mean'],
    'vendio_algo': ['mean'],
    'es_listado_free': ['mean'],
    'es_listado_gold': ['mean'],
    'tiene_garantia': ['mean'],
    'automatic_relist': ['mean'],
    'free_shipping': ['mean'],
    'foto_pct_camara': ['mean'],
    'foto_pct_cuadrada': ['mean'],
    'n_fotos': ['mean'],
    'n_atributos': ['mean'],
    'n_variaciones': ['mean'],
    'es_clasificado': ['mean'],
    'inicio_ts': ['min', 'max'],
}
VARIABLES_PERFIL_CATEGORIA = {
    'log_price': ['median', 'std'],
    'es_unidad_unica': ['mean'],
    'es_listado_free': ['mean'],
    'tiene_garantia': ['mean'],
    'foto_pct_camara': ['mean'],
}
# Variables que se comparan contra el promedio de su vendedor y de su categoría
VARIABLES_DESVIO = ['log_price', 'log_initial_quantity', 'foto_pct_camara', 'n_fotos', 'tiene_garantia', 'es_listado_free']


def variables_base(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'seller_id': X['seller_id'],
        'category_id': X['category_id'],
        'listing_type_id': X['listing_type_id'],
        'log_price': np.log1p(X['price']),
        'log_initial_quantity': np.log1p(X['initial_quantity']),
        'es_unidad_unica': (X['initial_quantity'] == 1).astype(int),
        'log_sold_quantity': np.log1p(X['sold_quantity']),
        'vendio_algo': (X['sold_quantity'] > 0).astype(int),
        'es_listado_free': (X['listing_type_id'] == 'free').astype(int),
        'es_listado_gold': X['listing_type_id'].str.startswith('gold').astype(int),
        'tiene_garantia': X['tiene_garantia'],
        'automatic_relist': X['automatic_relist'],
        'free_shipping': X['free_shipping'],
        'foto_pct_camara': X['foto_pct_camara'],
        'foto_pct_cuadrada': X['foto_pct_cuadrada'],
        'n_fotos': X['n_fotos'],
        'n_atributos': X['n_atributos'],
        'n_variaciones': X['n_variaciones'],
        'es_clasificado': (X['buying_mode'] == 'classified').astype(int),
        'inicio_ts': X['inicio_ts'],
    }, index=X.index)


def perfil_de_grupo(base: pd.DataFrame, columna_grupo: str, variables: dict, prefijo: str) -> pd.DataFrame:
    perfil = base.groupby(columna_grupo).agg(variables)
    perfil.columns = [f'{prefijo}_{variable}_{estadistico}' for variable, estadistico in perfil.columns]
    return base[[columna_grupo]].join(perfil, on=columna_grupo).drop(columns=columna_grupo)


def features_relacion_con_grupo(base: pd.DataFrame) -> pd.DataFrame:
    relacion = pd.DataFrame(index=base.index)
    for variable in VARIABLES_DESVIO:
        relacion[f'desvio_vendedor_{variable}'] = base[variable] - base.groupby('seller_id')[variable].transform('mean')
        relacion[f'desvio_categoria_{variable}'] = base[variable] - base.groupby('category_id')[variable].transform('mean')
    relacion['desvio_vendedor_x_categoria_log_price'] = base['log_price'] - base.groupby(['seller_id', 'category_id'])['log_price'].transform('median')

    tipo_mas_usado = base.groupby('seller_id')['listing_type_id'].agg(lambda tipos: tipos.value_counts().index[0])
    relacion['tipo_distinto_al_habitual'] = (base['listing_type_id'] != base['seller_id'].map(tipo_mas_usado)).astype(int)

    publicaciones_vendedor = base.groupby('seller_id')['seller_id'].transform('size')
    publicaciones_en_categoria = base.groupby(['seller_id', 'category_id'])['seller_id'].transform('size')
    relacion['pct_vendedor_en_esta_categoria'] = publicaciones_en_categoria / publicaciones_vendedor
    relacion['n_categorias_vendedor'] = base.groupby('seller_id')['category_id'].transform('nunique')
    relacion['ranking_precio_en_vendedor'] = base.groupby('seller_id')['log_price'].rank(pct=True)
    relacion['ranking_precio_en_categoria'] = base.groupby('category_id')['log_price'].rank(pct=True)
    return relacion


def agregar_features_perfil(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = len(X_train)
    base = variables_base(pd.concat([X_train, X_test], ignore_index=True))
    nuevas = pd.concat([
        perfil_de_grupo(base, 'seller_id', VARIABLES_PERFIL_VENDEDOR, 'vend'),
        perfil_de_grupo(base, 'category_id', VARIABLES_PERFIL_CATEGORIA, 'cat'),
        features_relacion_con_grupo(base),
    ], axis=1)
    nuevas['vend_antiguedad_dias'] = (nuevas.pop('vend_inicio_ts_max') - nuevas.pop('vend_inicio_ts_min')) / 86400
    X_train = pd.concat([X_train.reset_index(drop=True), nuevas.iloc[:n_train].reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), nuevas.iloc[n_train:].reset_index(drop=True)], axis=1)
    return X_train, X_test
