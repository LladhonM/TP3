"""
Features de "conocimiento del mundo" sobre título y garantía: patrones que una persona entiende y un conteo de palabras (TF-IDF) no.

Surgieron de leer a mano 250 casos difíciles de train (notebook 29). No usan el target: son reglas de texto que se calculan igual
para cualquier publicación, así que se pueden agregar a las matrices de la CV anidada sin recalcular encodings.
"""
import re
import unicodedata

import pandas as pd

# Cada regla: (nombre, campo sobre el que se busca, expresión regular). Los textos ya vienen en minúscula y sin acentos.
REGLAS_CONOCIMIENTO = [
    # Afirmación explícita de nuevo; excluye "como nuevo", "casi nuevo", "nueva edición", "nuevos y usados", etc.
    ('con_afirma_nuevo', 'titulo_garantia',
     r'(?<!como )(?<!casi )(?<!igual a )\bnuev[oa]s?\b(?!\s+(edicion|ed\b|coplas|version|era|testamento|mundo|york|temporada|orden|y usad))'
     r'|(?<!casi )(?<!poco )\bsin (uso|usar|abrir)\b|\ba estrenar\b|\bsellad[oa]\b|\b0 ?km\b|\bnunca (usad|se uso)'),
    # Afirmación explícita de usado o reacondicionado; excluye "nuevos y usados" y "ni usado"
    ('con_afirma_usado', 'titulo_garantia',
     r'(?<!y )(?<!ni )(?<!no haya sido )\busad[oa]s?\b|\bcasi sin uso\b|\bcomo nuev[oa]\b'
     r'|\b(excelente|exelente|muy buen|buen|perfecto) estado\b|\bexcel\.? est\b|\bpara reparar\b|\ba reparar\b|\bpera reparar\b'
     r'|\brepuestos?\b|\broto\b|\brefabricad|\breacondicionad|\bprobados\b'),
    # Garantía basada en la reputación del vendedor y no en el fabricante → particular
    ('con_garantia_reputacion', 'garantia', r'calificaci|reputaci|trayectoria'),
    # Servicios, alquileres, entradas, vouchers: no pueden ser "usados"
    ('con_intangible', 'titulo',
     r'\bservicio\b(?! de (te|cafe|mesa|vajilla|living))|\balquiler\b|\bentradas?\b|\bcards?\s+\d+\s*(u\$|usd|dolar|peso|\$)'
     r'|\bpsn\b|\bsteam\b|\bcodigo\b|\brecarga\b|orden de compra|\bstud\b|\bcupon\b|\bvoucher\b|\bpasajes?\b'),
    # Packs, docenas, mayorista → comercio
    ('con_pack_mayorista', 'titulo',
     r'\bpack\b|\bdocena\b|descartable|\b(por|x) mayor\b|\bmayorista\b|\bx ?(50|100|200|500|1000) ?(u|un|unid)\b|por unidad|\bcombo\b'),
    # Réplicas a escala y fascículos de colección → producto comercial nuevo aunque diga "de época"
    ('con_replica_escala', 'titulo', r'\baltaya\b|\bital?eri\b|\bitalieri\b|\b1/(18|24|32|35|43|48|72)\b|\bescala\b'),
    # "Lote de..." → particular vaciando cosas
    ('con_lote', 'titulo', r'\blote\b|\bvari[oa]s\b(?! (colores|talles|modelos|disenos))'),
    # Copias digitales (PDF, software) → se publican como usadas
    ('con_digital', 'titulo', r'libro digital|\be-?books?\b|\bpdf\b|\bphotoshop\b|\bcubase\b|\bautocad\b|\btutoriales?\b'),
    # Obra de arte original → nueva
    ('con_obra_arte', 'titulo', r'\bcuadros?\b(?! afiche)|\bescultura\b|\boleo\b|\btinta china\b|\bacuarela\b'),
]
COLUMNAS_CONOCIMIENTO = [nombre for nombre, _, _ in REGLAS_CONOCIMIENTO]


def normalizar(texto: str) -> str:
    if not isinstance(texto, str):
        return ''
    sin_acentos = unicodedata.normalize('NFKD', texto.lower())
    return ''.join(caracter for caracter in sin_acentos if not unicodedata.combining(caracter))


def features_conocimiento(T: pd.DataFrame, reglas=REGLAS_CONOCIMIENTO) -> pd.DataFrame:
    """T es la tabla de textos de construir_dataset() (columnas titulo, garantia)."""
    titulo = T['titulo'].map(normalizar)
    garantia = T['garantia'].replace('sin_dato', '').map(normalizar)
    campos = {'titulo': titulo, 'garantia': garantia, 'titulo_garantia': titulo + ' | ' + garantia}
    columnas = {}
    for nombre, campo, patron in reglas:
        expresion = re.compile(patron)
        columnas[nombre] = campos[campo].map(lambda texto: int(bool(expresion.search(texto))))
    return pd.DataFrame(columnas, index=T.index).reset_index(drop=True)
