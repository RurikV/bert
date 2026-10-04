"""Исполняет ноутбук, сохраняя его после каждой выполненной ячейки.

Запуск из корня проекта:
    .venv/bin/python scripts/run_notebook.py hw_bert_rucola.ipynb
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

nb_path = Path(sys.argv[1] if len(sys.argv) > 1 else "hw_bert_rucola.ipynb")
nb = nbformat.read(nb_path, as_version=4)

client = NotebookClient(
    nb,
    timeout=6 * 3600,  # суммарный бюджет на весь ноутбук, сек
    kernel_name="hw-bert",
    resources={"metadata": {"path": str(Path.cwd())}},
)

t0 = time.time()
n_cells = len(nb.cells)


def on_cell_executed(cell=None, cell_index=None, **kwargs):
    first_line = next((l for l in (cell.source if cell else "").splitlines() if l.strip()), "")
    print(f"[{time.time() - t0:8.0f}s] готова ячейка {cell_index + 1}/{n_cells}: {first_line[:70]}",
          flush=True)
    nbformat.write(nb, nb_path)  # сохраняем прогресс после каждой ячейки


client.on_cell_executed = on_cell_executed

try:
    client.execute()
finally:
    nbformat.write(nb, nb_path)
    print(f"[{time.time() - t0:8.0f}s] ноутбук сохранён: {nb_path}", flush=True)
