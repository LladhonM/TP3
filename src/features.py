"""Aplanado del JSON de MercadoLibre y features estructuradas (sin target)."""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# np.nanmax/nanmean avisan cuando todas las fotos vienen sin max_size; el NaN resultante es correcto
warnings.filterwarnings('ignore', category=RuntimeWarning)

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / 'data'
RUTA_PROCESADOS = RUTA_DATOS / 'processed'

TAGS_CONOCIDOS = [
    'dragged_bids_and_visits', 'good_quality_thumbnail', 'dragged_visits',
    'free_relist', 'poor_quality_thumbnail',
]
COLUMNAS_CATEGORICAS = [
    'listing_type_id', 'buying_mode', 'status', 'sub_status', 'currency_id',
    'shipping_mode', 'state_id', 'city_id', 'category_id', 'seller_id',
    'primera_foto_max_size', 'formato_id_foto',
]


def cargar_crudos() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_json(RUTA_DATOS / 'train_data.jsonlines', lines=True)
    test = pd.read_json(RUTA_DATOS / 'test_data.jsonlines', lines=True)
    return train, test


def parse_dimensiones(texto: str) -> tuple[float, float]:
    if not texto or 'x' not in texto:
        return np.nan, np.nan
    ancho, alto = texto.split('x')
    return float(ancho), float(alto)


def features_fotos(fotos: list[dict], id_item: str) -> dict:
    if not fotos:
        return {'n_fotos': 0}
    anchos, altos, proporciones = [], [], []
    for foto in fotos:
        ancho, alto = parse_dimensiones(foto.get('max_size', ''))
        anchos.append(ancho)
        altos.append(alto)
        proporciones.append(ancho / alto if alto else np.nan)
    proporciones = np.array(proporciones)
    # Proporción 4:3 / 3:4 es típica de cámara de celular (particulares → usado)
    es_camara = np.isclose(proporciones, 4 / 3, atol=0.02) | np.isclose(proporciones, 3 / 4, atol=0.02)
    es_cuadrada = np.isclose(proporciones, 1.0, atol=0.01)
    primera = fotos[0]
    sufijo = re.search(r'_(\d+)$', primera.get('id', ''))
    sufijo = sufijo.group(1) if sufijo else ''
    return {
        'n_fotos': len(fotos),
        'foto_ancho_max': np.nanmax(anchos),
        'foto_alto_max': np.nanmax(altos),
        'foto_area_media': np.nanmean(np.array(anchos) * np.array(altos)),
        'foto_proporcion_primera': proporciones[0],
        'foto_pct_camara': es_camara.mean(),
        'foto_pct_cuadrada': es_cuadrada.mean(),
        'foto_pct_size_igual_max': np.mean([f.get('size') == f.get('max_size') for f in fotos]),
        'foto_pct_id_propio': np.mean([id_item[3:] in f.get('id', '') for f in fotos]),
        'primera_foto_max_size': primera.get('max_size', ''),
        'formato_id_foto': re.sub(r'\d', '9', primera.get('id', '')),
        'foto_largo_sufijo': len(sufijo),
        # Sufijo MMYYYY = fecha de subida de la foto
        'foto_anio': float(sufijo[2:]) if len(sufijo) == 6 else np.nan,
        'foto_mes_abs': float(sufijo[2:]) * 12 + float(sufijo[:2]) if len(sufijo) == 6 else np.nan,
    }


def features_pagos(metodos: list[dict]) -> dict:
    ids = {metodo['id'] for metodo in metodos}
    return {f'pago_{codigo}': int(codigo in ids) for codigo in
            ['MLATB', 'MLAWC', 'MLAMO', 'MLAOT', 'MLAWT', 'MLAVE', 'MLAMC', 'MLAAM', 'MLAVS', 'MLADC', 'MLAMS', 'MLABC', 'MLAMP', 'MLACD']}


def aplanar(df: pd.DataFrame) -> pd.DataFrame:
    plano = pd.DataFrame(index=df.index)
    plano['id'] = df['id']

    # Categóricas directas
    plano['listing_type_id'] = df['listing_type_id']
    plano['buying_mode'] = df['buying_mode']
    plano['status'] = df['status']
    plano['sub_status'] = df['sub_status'].map(lambda l: l[0] if l else 'none')
    plano['currency_id'] = df['currency_id']
    plano['category_id'] = df['category_id']
    plano['seller_id'] = df['seller_id'].astype(str)
    plano['state_id'] = df['seller_address'].map(lambda d: d['state']['id'])
    plano['city_id'] = df['seller_address'].map(lambda d: d['city']['id'])

    # Precio y cantidades
    plano['price'] = df['price']
    plano['log_price'] = np.log1p(df['price'])
    plano['precio_redondo'] = (df['price'] % 100 == 0).astype(int)
    plano['precio_decimales'] = (df['price'] % 1 != 0).astype(int)
    plano['price_distinto_base'] = (df['price'] != df['base_price']).astype(int)
    plano['tiene_original_price'] = df['original_price'].notna().astype(int)
    plano['initial_quantity'] = df['initial_quantity']
    plano['sold_quantity'] = df['sold_quantity']
    plano['available_quantity'] = df['available_quantity']
    plano['ratio_vendido'] = df['sold_quantity'] / df['initial_quantity'].clip(lower=1)
    plano['diferencia_stock'] = df['initial_quantity'] - df['available_quantity'] - df['sold_quantity']

    # Flags
    plano['accepts_mercadopago'] = df['accepts_mercadopago'].astype(int)
    plano['automatic_relist'] = df['automatic_relist'].astype(int)
    plano['tiene_tienda_oficial'] = df['official_store_id'].notna().astype(int)
    plano['tiene_video'] = df['video_id'].notna().astype(int)
    plano['tiene_parent_item'] = df['parent_item_id'].notna().astype(int)
    plano['tiene_catalogo'] = df['catalog_product_id'].notna().astype(int)
    plano['n_deals'] = df['deal_ids'].map(len)
    plano['n_descripciones'] = df['descriptions'].map(len)

    # Envío
    plano['shipping_mode'] = df['shipping'].map(lambda d: d.get('mode'))
    plano['free_shipping'] = df['shipping'].map(lambda d: d.get('free_shipping')).astype(int)
    plano['local_pick_up'] = df['shipping'].map(lambda d: d.get('local_pick_up')).astype(int)
    plano['n_shipping_methods'] = df['shipping'].map(lambda d: len(d.get('methods') or []))
    plano['n_shipping_tags'] = df['shipping'].map(lambda d: len(d.get('tags') or []))
    plano['tiene_dimensiones'] = df['shipping'].map(lambda d: d.get('dimensions') is not None).astype(int)

    # Tags, variaciones, atributos
    for tag in TAGS_CONOCIDOS:
        plano[f'tag_{tag}'] = df['tags'].map(lambda l, t=tag: int(t in l))
    plano['n_variaciones'] = df['variations'].map(len)
    plano['variaciones_cantidad_total'] = df['variations'].map(lambda l: sum(v.get('available_quantity', 0) for v in l))
    plano['n_atributos'] = df['attributes'].map(len)

    # Pagos
    plano['n_pagos_no_mp'] = df['non_mercado_pago_payment_methods'].map(len)
    plano = plano.join(pd.DataFrame(list(df['non_mercado_pago_payment_methods'].map(features_pagos)), index=df.index))

    # Fotos
    fotos = [features_fotos(f, i) for f, i in zip(df['pictures'], df['id'])]
    plano = plano.join(pd.DataFrame(fotos, index=df.index))

    # Fechas
    inicio = pd.to_datetime(df['start_time'])
    fin = pd.to_datetime(df['stop_time'])
    actualizado = pd.to_datetime(df['last_updated']).dt.tz_localize(None)
    creado = pd.to_datetime(df['date_created']).dt.tz_localize(None)
    plano['inicio_ts'] = inicio.astype('int64') // 10**9
    plano['inicio_mes'] = inicio.dt.month
    plano['inicio_dia_semana'] = inicio.dt.dayofweek
    plano['inicio_hora'] = inicio.dt.hour
    plano['duracion_dias'] = (fin - inicio).dt.total_seconds() / 86400
    plano['dias_hasta_actualizacion'] = (actualizado - creado).dt.total_seconds() / 86400
    plano['inicio_mes_abs'] = inicio.dt.year * 12 + inicio.dt.month
    plano['meses_foto_a_inicio'] = plano['inicio_mes_abs'] - plano['foto_mes_abs']

    # Texto (longitudes; el contenido va a los modelos de texto)
    titulo = df['title'].fillna('')
    garantia = df['warranty'].fillna('')
    plano['tiene_garantia'] = df['warranty'].notna().astype(int)
    plano['largo_titulo'] = titulo.str.len()
    plano['palabras_titulo'] = titulo.str.split().map(len)
    plano['pct_mayusculas_titulo'] = titulo.map(lambda t: sum(c.isupper() for c in t) / max(len(t), 1))
    plano['digitos_titulo'] = titulo.str.count(r'\d')
    plano['largo_garantia'] = garantia.str.len()
    return plano


def textos_para_modelo(df: pd.DataFrame) -> pd.DataFrame:
    atributos = df['attributes'].map(lambda l: ' '.join(f"{a.get('name', '')} {a.get('value_name', '')}" for a in l))
    return pd.DataFrame({
        'titulo': df['title'].fillna(''),
        'garantia': df['warranty'].fillna('sin_dato'),
        'atributos': atributos,
    }, index=df.index)


def construir_dataset(usar_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Devuelve (features_train, features_test, target, textos_train, textos_test)."""
    RUTA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    rutas = {nombre: RUTA_PROCESADOS / f'{nombre}.pkl' for nombre in ['X_train', 'X_test', 'y', 'T_train', 'T_test']}
    if usar_cache and all(ruta.exists() for ruta in rutas.values()):
        return tuple(pd.read_pickle(rutas[nombre]) for nombre in rutas)

    train, test = cargar_crudos()
    target = (train['condition'] == 'used').astype(int).rename('es_usado')
    resultado = {
        'X_train': aplanar(train), 'X_test': aplanar(test), 'y': target,
        'T_train': textos_para_modelo(train), 'T_test': textos_para_modelo(test),
    }
    for nombre, objeto in resultado.items():
        objeto.to_pickle(rutas[nombre])
    return tuple(resultado.values())
