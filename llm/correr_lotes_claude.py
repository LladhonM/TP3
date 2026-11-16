"""Corre lotes del LLM dirigido con Claude Fable 5.2 vía Claude Code CLI (`claude -p`), corrida `fable`.

Mismo esquema que `correr_lotes_codex.py` y `correr_lotes_agy.py`: cada lote corre en una carpeta temporal aislada que solo
tiene la guía v2 congelada (`guia.md`) y el lote (`lote.csv`), con el mismo prompt. La salida se valida y se copia a
`data/llm/resultados/<version>_<lote>.json`, el mismo formato que las corridas de Opus.

Aislamiento: `--setting-sources ""` (sin CLAUDE.md, memoria, hooks ni plugins; `--bare` no sirve porque saltea el login),
sin MCP, sin skills, solo las herramientas Read y Write, y la carpeta aislada como directorio de trabajo.

Uso (desde la raíz del proyecto):
    python3 llm/correr_lotes_claude.py --version fable --lotes A B C V1 V2 V3 V4
    python3 llm/correr_lotes_claude.py --version fable --lotes A --prueba 5   # prueba de humo, no guarda
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

CLAUDE = shutil.which('claude') or 'claude'
CARPETA = Path('data/llm')
GUIA = Path('llm/guia_llm_dirigido_v2.md')
PLANTILLA = Path('llm/prompt_llm_codex.md')
LOGS = CARPETA / 'resultados' / 'logs_claude'


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


def resumir_uso(stdout: str) -> str:
    """Tokens y costo informados por el CLI (con un modelo que el CLI no conoce, el costo puede ser aproximado)."""
    try:
        datos = json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return 'uso no disponible'
    uso = datos.get('usage', {})
    entrada = (uso.get('input_tokens', 0) + uso.get('cache_creation_input_tokens', 0)
               + uso.get('cache_read_input_tokens', 0))
    return (f'entrada {entrada:,} tok, salida {uso.get("output_tokens", 0):,} tok, '
            f'${datos.get("total_cost_usd", 0):.2f}, modelos {list((datos.get("modelUsage") or {}).keys())}')


def correr_lote(nombre_lote: str, version: str, modelo: str, esfuerzo: str, prueba: int | None) -> str:
    lote = pd.read_csv(CARPETA / 'lotes' / f'lote_{nombre_lote}.csv')
    if prueba:
        lote = lote.head(prueba)
    carpeta_aislada = Path(tempfile.mkdtemp(prefix='lote_'))
    shutil.copyfile(GUIA, carpeta_aislada / 'guia.md')
    lote.to_csv(carpeta_aislada / 'lote.csv', index=False)

    # Solo Read y Write; las ediciones se aprueban solas para poder escribir resultado.json.
    comando = [CLAUDE, '-p', armar_prompt(lote), '--model', modelo, '--effort', esfuerzo,
               '--setting-sources', '', '--strict-mcp-config', '--disable-slash-commands', '--no-session-persistence',
               '--tools', 'Read,Write', '--permission-mode', 'acceptEdits', '--output-format', 'json']
    inicio = time.time()
    proceso = subprocess.run(comando, capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=carpeta_aislada)
    minutos = (time.time() - inicio) / 60
    uso = resumir_uso(proceso.stdout)

    LOGS.mkdir(parents=True, exist_ok=True)
    sufijo = f'_prueba{prueba}' if prueba else ''
    (LOGS / f'{version}_{nombre_lote}{sufijo}.log').write_text(proceso.stdout + '\n--- stderr ---\n' + proceso.stderr)

    try:
        respuestas = validar(carpeta_aislada / 'resultado.json', lote.id.tolist())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        return f'{nombre_lote}: ERROR ({error}) en {minutos:.1f} min; código {proceso.returncode}; {uso}; ver log'
    finally:
        shutil.rmtree(carpeta_aislada, ignore_errors=True)
    if prueba:
        return f'{nombre_lote}: prueba OK, {len(respuestas)} ids en {minutos:.1f} min; {uso}: {respuestas}'
    destino = CARPETA / 'resultados' / f'{version}_{nombre_lote}.json'
    destino.write_text(json.dumps(respuestas, ensure_ascii=False, indent=1))
    return f'{nombre_lote}: OK, {len(respuestas)} ids en {minutos:.1f} min; {uso} → {destino.name}'


def main():
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument('--version', required=True)
    argumentos.add_argument('--lotes', nargs='+', required=True)
    argumentos.add_argument('--modelo', default='claude-fable-5-2')
    argumentos.add_argument('--esfuerzo', default='high')
    argumentos.add_argument('--paralelo', type=int, default=4)
    argumentos.add_argument('--prueba', type=int, default=None, help='solo las primeras N filas; no guarda resultados')
    args = argumentos.parse_args()

    pendientes = [lote for lote in args.lotes
                  if args.prueba or not (CARPETA / 'resultados' / f'{args.version}_{lote}.json').exists()]
    print(f'{len(pendientes)} lotes a correr con {args.modelo} (esfuerzo {args.esfuerzo}): {pendientes}', flush=True)
    errores = 0
    with ThreadPoolExecutor(args.paralelo) as ejecutor:
        for mensaje in ejecutor.map(lambda lote: correr_lote(lote, args.version, args.modelo, args.esfuerzo, args.prueba),
                                    pendientes):
            print(mensaje, flush=True)
            errores += 'ERROR' in mensaje
    # Código de salida distinto de cero si falló algún lote, para que `&&` no siga con la evaluación.
    if errores:
        sys.exit(f'{errores} lotes con error; volver a correr el mismo comando (saltea los que ya están)')


if __name__ == '__main__':
    main()
