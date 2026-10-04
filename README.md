# Дообучение BERT на RuCoLA 

Основной артефакт — **`hw_bert_rucola.ipynb`**: выполненный ноутбук с кодом, результатами
и markdown-анализом по всем пунктам задания (данные → токенизация → обучение ruBert-base →
оценка и анализ ошибок → rubert-tiny2 и сравнение → подбор гиперпараметров).

## Воспроизведение

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name hw-bert --display-name hw-bert

# пересобрать «чистый» ноутбук из scripts/build_notebook.py и выполнить его целиком
.venv/bin/python scripts/build_notebook.py
.venv/bin/python scripts/run_notebook.py hw_bert_rucola.ipynb
.venv/bin/python scripts/fill_analysis.py   # вписать аналитические markdown-ячейки
```

Данные RuCoLA скачиваются автоматически в `data/`, предобученные модели — из Hugging Face Hub.
Полное выполнение на Apple M1 Max (mps) занимает ~65 минут.

## Структура

- `hw_bert_rucola.ipynb` — Основной артефакт
- `scripts/build_notebook.py` — генератор ноутбука
- `scripts/run_notebook.py` — исполнитель с сохранением после каждой ячейки
- `scripts/fill_analysis.py` — вставка аналитических выводов по фактическим результатам
- `scripts/smoke_test.py` — быстрая проверка API Trainer перед полным запуском
- `logs/metrics.jsonl`, `logs/grid_progress.csv` — метрики всех запусков
- `outputs/` — чекпойнты (`data/`, `outputs/`, `logs/` в `.gitignore`)
