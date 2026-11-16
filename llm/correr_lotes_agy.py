"""Corre lotes del LLM dirigido con Gemini 4 Argon vía Antigravity CLI (`agy`), corrida `argon`.

Mismo esquema que `correr_lotes_codex.py`: cada lote corre en una carpeta temporal aislada que solo tiene la guía v2
congelada (`guia.md`) y el lote (`lote.csv`), con el mismo prompt. La salida se valida y se copia a
`data/llm/resultados/<version>_<lote>.json`, el mismo formato que las corridas de Opus.

Uso (desde la raíz del proyecto):
    python3 llm/correr_lotes_agy.py --version argon --lotes A B C V1 V2 V3 V4
    python3 llm/correr_lotes_agy.py --version argon --lotes A --prueba 5   # prueba de humo, no guarda
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

AGY = '/Applications/Gemini.app/Contents/Helpers/agy'
CARPETA = Path('data/llm')
GUIA = Path('llm/guia_llm_dirigido_v2.md')
PLANTILLA = Path('llm/prompt_llm_codex.md')
LOGS = CARPETA / 'resultados' / 'logs_agy'


def armar_prompt(lote: pd.DataFrame) -> str:
    return (PLANTILLA.read_text()
            .replace('<N>', str(len(lote)))
            .replace('<FIRST_ID>', str(lote.id.iloc[0]))
            .replace('<LAST_ID>', str(lote.id.iloc[-1])))


def validar(salida: Path, ids: list[int]) -> dict:
    """Devuelve las respuestas si están todos los ids con p entre 0 y 1; si no, levanta un error."""
    respuestas = json.loads(salida.read_text())
    faltantes = {str(i) for i in ids} - set(respuestas)
    if faltantes:
        raise ValueError(f'faltan {len(faltantes)} ids')
    for id_fila, respuesta in respuestas.items():
        if not 0 <= float(respuesta['p']) <= 1:
            raise ValueError(f'p fuera de rango en el id {id_fila}')
    return respuestas


def correr_lote(nombre_lote: str, version: str, modelo: str, prueba: int | None) -> str:
    lote = pd.read_csv(CARPETA / 'lotes' / f'lote_{nombre_lote}.csv')
    if prueba:
        lote = lote.head(prueba)
    carpeta_aislada = Path(tempfile.mkdtemp(prefix='lote_'))
    shutil.copyfile(GUIA, carpeta_aislada / 'guia.md')
    lote.to_csv(carpeta_aislada / 'lote.csv', index=False)

    # Sin terminal libre (--sandbox) y sin skills; solo se aprueban solas las ediciones de archivos (para escribir resultado.json).
    comando = [AGY, '-p', armar_prompt(lote), '--model', modelo, '--sandbox', '--disable-slash-commands',
               '--mode', 'accept-edits', '--print-timeout', '45m']
    inicio = time.time()
    proceso = subprocess.run(comando, capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=carpeta_aislada)
    minutos = (time.time() - inicio) / 60

    LOGS.mkdir(parents=True, exist_ok=True)
    sufijo = f'_prueba{prueba}' if prueba else ''
    (LOGS / f'{version}_{nombre_lote}{sufijo}.log').write_text(proceso.stdout + '\n--- stderr ---\n' + proceso.stderr)

    try:
        respuestas = validar(carpeta_aislada / 'resultado.json', lote.id.tolist())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        return f'{nombre_lote}: ERROR ({error}) en {minutos:.1f} min; código {proceso.returncode}; ver log'
    finally:
        shutil.rmtree(carpeta_aislada, ignore_errors=True)
    if prueba:
        return f'{nombre_lote}: prueba OK, {len(respuestas)} ids en {minutos:.1f} min: {respuestas}'
    destino = CARPETA / 'resultados' / f'{version}_{nombre_lote}.json'
    destino.write_text(json.dumps(respuestas, ensure_ascii=False, indent=1))
    return f'{nombre_lote}: OK, {len(respuestas)} ids en {minutos:.1f} min → {destino.name}'


def main():
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument('--version', required=True)
    argumentos.add_argument('--lotes', nargs='+', required=True)
    argumentos.add_argument('--modelo', default='gemini-4-argon-high')
    argumentos.add_argument('--paralelo', type=int, default=8)
    argumentos.add_argument('--prueba', type=int, default=None, help='solo las primeras N filas; no guarda resultados')
    args = argumentos.parse_args()

    pendientes = [lote for lote in args.lotes
                  if args.prueba or not (CARPETA / 'resultados' / f'{args.version}_{lote}.json').exists()]
    print(f'{len(pendientes)} lotes a correr con {args.modelo}: {pendientes}', flush=True)
    errores = 0
    with ThreadPoolExecutor(args.paralelo) as ejecutor:
        for mensaje in ejecutor.map(lambda lote: correr_lote(lote, args.version, args.modelo, args.prueba), pendientes):
            print(mensaje, flush=True)
            errores += 'ERROR' in mensaje
    # Código de salida distinto de cero si falló algún lote, para que `&&` no siga con la evaluación.
    if errores:
        sys.exit(f'{errores} lotes con error; volver a correr el mismo comando (saltea los que ya están)')


if __name__ == '__main__':
    main()
