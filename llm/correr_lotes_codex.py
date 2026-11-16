"""Corre lotes del LLM dirigido con un modelo de OpenAI vía Codex CLI (segundo LLM, corrida `sol`).

Cada lote corre en una carpeta temporal aislada que solo tiene la guía v2 congelada (`guia.md`) y el lote (`lote.csv`):
el modelo no ve la ruta del proyecto ni ningún otro archivo. La salida se valida y se copia a
`data/llm/resultados/<version>_<lote>.json`, el mismo formato que las corridas de Opus.

Uso (desde la raíz del proyecto):
    python3 llm/correr_lotes_codex.py --version sol --lotes A B C V1 V2 V3 V4
    python3 llm/correr_lotes_codex.py --version sol --lotes A --prueba 5   # prueba de humo, no guarda
"""
import argparse
import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

CODEX = '/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex'
CARPETA = Path('data/llm')
GUIA = Path('llm/guia_llm_dirigido_v2.md')
PLANTILLA = Path('llm/prompt_llm_codex.md')
LOGS = CARPETA / 'resultados' / 'logs_codex'


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


def correr_lote(nombre_lote: str, version: str, modelo: str, esfuerzo: str, prueba: int | None) -> str:
    lote = pd.read_csv(CARPETA / 'lotes' / f'lote_{nombre_lote}.csv')
    if prueba:
        lote = lote.head(prueba)
    carpeta_aislada = Path(tempfile.mkdtemp(prefix='lote_'))
    shutil.copyfile(GUIA, carpeta_aislada / 'guia.md')
    lote.to_csv(carpeta_aislada / 'lote.csv', index=False)

    comando = [CODEX, 'exec', '-m', modelo, '-c', f'model_reasoning_effort="{esfuerzo}"',
               '--ignore-user-config', '--skip-git-repo-check', '--ephemeral',
               '-s', 'workspace-write', '-C', str(carpeta_aislada), armar_prompt(lote)]
    inicio = time.time()
    proceso = subprocess.run(comando, capture_output=True, text=True, stdin=subprocess.DEVNULL)
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
    argumentos.add_argument('--modelo', default='gpt-6.1-sol')
    argumentos.add_argument('--esfuerzo', default='high')
    argumentos.add_argument('--paralelo', type=int, default=8)
    argumentos.add_argument('--prueba', type=int, default=None, help='solo las primeras N filas; no guarda resultados')
    args = argumentos.parse_args()

    pendientes = [lote for lote in args.lotes
                  if args.prueba or not (CARPETA / 'resultados' / f'{args.version}_{lote}.json').exists()]
    print(f'{len(pendientes)} lotes a correr con {args.modelo} (esfuerzo {args.esfuerzo}): {pendientes}', flush=True)
    with ThreadPoolExecutor(args.paralelo) as ejecutor:
        for mensaje in ejecutor.map(lambda lote: correr_lote(lote, args.version, args.modelo, args.esfuerzo, args.prueba),
                                    pendientes):
            print(mensaje, flush=True)


if __name__ == '__main__':
    main()
