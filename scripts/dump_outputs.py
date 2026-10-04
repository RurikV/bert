"""Печатает текстовые выводы всех ячеек ноутбука: python scripts/dump_outputs.py"""
import sys

import nbformat

nb = nbformat.read(sys.argv[1] if len(sys.argv) > 1 else "hw_bert_rucola.ipynb", as_version=4)
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    texts = []
    for out in cell.get("outputs", []):
        if out.output_type == "stream":
            texts.append(out.text)
        elif out.output_type == "execute_result":
            data = out.get("data", {})
            if "text/plain" in data:
                texts.append(data["text/plain"])
        elif out.output_type == "error":
            texts.append(f"ERROR: {out.ename}: {out.evalue}")
    joined = "".join(texts).strip()
    if not joined:
        continue
    first = next((l for l in cell.source.splitlines() if l.strip()), "")
    print(f"\n===== cell {i}: {first[:60]} =====")
    print(joined[:6000])
