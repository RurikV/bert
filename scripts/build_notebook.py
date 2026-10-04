"""Генерирует hw_bert_rucola.ipynb из списка ячеек.

Запуск:  .venv/bin/python scripts/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

# ============================== шапка ==============================
cells.append(md("""\
# Дообучение BERT на RuCoLA

**Задача** — бинарная классификация: приемлемо ли предложение русского языка с точки зрения грамматики
(RuCoLA — Russian Corpus of Linguistic Acceptability, аналог английской CoLA).

План работы:
1. Окружение и данные (`in_domain_train.csv` → train/val 80/20, `in_domain_dev.csv` → test)
2. Токенизация (`datasets` + токенизатор `sberbank-ai/ruBert-base`)
3. Обучение `sberbank-ai/ruBert-base`
4. Оценка качества на test + анализ ошибок
5. Обучение `cointegrated/rubert-tiny2` + сравнение моделей
6. Подбор гиперпараметров (learning rate, batch size, warmup ratio)

Оборудование: Apple M1 Max, устройство `mps` (Metal Performance Shaders).
"""))

cells.append(code("""\
import itertools
import json
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# на Mac с Apple Silicon доступен ускоритель mps
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"transformers {transformers.__version__} | torch {torch.__version__} | device: {DEVICE}")
"""))

# ============================== 1. данные ==============================
cells.append(md("""\
## 1. Окружение и данные

Установка: `pip install transformers datasets accelerate` (плюс `torch`, `scikit-learn`, `pandas`).

Датасет [RuCoLA](https://github.com/RussianNLP/RuCoLA):
- `in_domain_train.csv` — обучающая выборка, внутри делим её на **train (80%)** и **val (20%)** со стратификацией по классу;
- `in_domain_dev.csv` — выступает в роли **test**.

Оставляем только столбцы `sentence` и `acceptable` (целевая переменная: 1 — грамматически приемлемое предложение, 0 — ошибочное).
"""))

cells.append(code("""\
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RUCOLA_URLS = {
    "in_domain_train.csv": "https://raw.githubusercontent.com/RussianNLP/RuCoLA/main/data/in_domain_train.csv",
    "in_domain_dev.csv": "https://raw.githubusercontent.com/RussianNLP/RuCoLA/main/data/in_domain_dev.csv",
}
for name, url in RUCOLA_URLS.items():  # скачиваем один раз
    if not (DATA_DIR / name).exists():
        urllib.request.urlretrieve(url, DATA_DIR / name)

full_train = pd.read_csv(DATA_DIR / "in_domain_train.csv")
test_df = pd.read_csv(DATA_DIR / "in_domain_dev.csv")

# оставляем только нужные столбцы: sentence и acceptable (целевая)
full_train = full_train[["sentence", "acceptable"]]
test_df = test_df[["sentence", "acceptable"]]

# train (80%) / val (20%): стратификация по целевой, чтобы сохранить баланс классов
train_df, val_df = train_test_split(
    full_train, test_size=0.2, stratify=full_train["acceptable"], random_state=SEED
)

print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name}: доля приемлемых предложений = {df['acceptable'].mean():.2%}")
"""))

# ============================== 2. токенизация ==============================
cells.append(md("""\
## 2. Токенизация данных

Переводим выборки в формат `datasets.DatasetDict` и токенизируем токенизатором `sberbank-ai/ruBert-base`.
Целевую колонку переименовываем в `label` — именно её ждёт `AutoModelForSequenceClassification`.
"""))

cells.append(code("""\
BASE_MODEL = "sberbank-ai/ruBert-base"
TINY_MODEL = "cointegrated/rubert-tiny2"

# прикидываем длину предложений в токенах, чтобы выбрать max_length
tmp_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
lens = [len(ids) for ids in tmp_tok(list(full_train["sentence"]))["input_ids"]]
print("длина в токенах, перцентили 50/95/99/100:",
      np.percentile(lens, [50, 95, 99, 100]).round(1))
"""))

cells.append(code("""\
MAX_LEN = 64  # с запасом покрывает все предложения (99-й перцентиль << 64)


def build_datasets(model_name: str):
    \"\"\"Токенизирует train/val/test токенизатором указенной модели.\"\"\"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "val": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    }).rename_column("acceptable", "label")  # стандартное имя целевой колонки
    ds = ds.map(
        lambda batch: tokenizer(batch["sentence"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["sentence"],
    )
    return tokenizer, ds


base_tokenizer, base_ds = build_datasets(BASE_MODEL)
print(base_ds)
print(base_ds["train"][0])
"""))

# ============================== 3. обучение ruBert-base ==============================
cells.append(md("""\
## 3. Обучение `sberbank-ai/ruBert-base`

Гиперпараметры (в рамках, заданных заданием):
- оптимизатор **AdamW** (`adamw_torch` — оптимизатор `Trainer` по умолчанию);
- `batch_size = 16` (диапазон 8–32);
- `learning_rate = 2e-5` (диапазон 1e-5 — 5e-5);
- до 4 эпох, оценка на val после каждой эпохи;
- обучение продолжаем, пока val loss не выйдет на плато: `EarlyStoppingCallback` с patience = 2
  (останавливаемся после двух эпох подряд без улучшения val loss),
  лучшая по val loss модель восстанавливается в конце (`load_best_model_at_end`).

Для контроля времени каждая эпоха замеряется колбэком `RunTracker`; метрики дублируются в `logs/metrics.jsonl`.
"""))

cells.append(code("""\
LOG_FILE = Path("logs/metrics.jsonl")  # дублируем метрики сюда для мониторинга
LOG_FILE.parent.mkdir(exist_ok=True)


def compute_metrics(eval_pred):
    \"\"\"Accuracy, F1 и MCC по logits.\"\"\"
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "mcc": matthews_corrcoef(labels, preds),
    }


class RunTracker(TrainerCallback):
    \"\"\"Замеряет длительность каждой эпохи; метрики валидации добирает из on_log
    (в on_epoch_end они ещё не готовы) и пишет их в jsonl-лог.
    После trainer.train() сбор отключается (tracker.capture = False), чтобы финальный
    evaluate() лучшей модели не затирал метрики последней эпохи.\"\"\"

    def __init__(self, run_name: str):
        self.run_name = run_name
        self._t0 = None
        self.capture = True
        self.history = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._t0 = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.history.append({
            "run": self.run_name,
            "epoch": round(state.epoch, 2),
            "epoch_time_s": round(time.time() - self._t0, 1),
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not self.history or not self.capture:
            return
        rec = self.history[-1]
        if "eval_loss" in logs:  # метрики валидации закончившейся эпохи
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    rec[key.removeprefix("eval_")] = round(float(value), 4)
            with LOG_FILE.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
        elif "loss" in logs:  # train loss эпохи
            rec["train_loss"] = round(float(logs["loss"]), 4)


def train_model(model_name, train_ds, eval_ds, *, run_name, lr=2e-5, batch_size=16,
                epochs=4, warmup_ratio=0.1, weight_decay=0.01, patience=2):
    \"\"\"Обучает модель классификации; возвращает (trainer, summary с таймингами и метриками).\"\"\"
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "unacceptable", 1: "acceptable"},
        label2id={"unacceptable": 0, "acceptable": 1},
    )

    args = TrainingArguments(
        output_dir=f"outputs/{run_name}",
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        warmup_ratio=warmup_ratio,        # разогрев lr на первых шагах
        weight_decay=weight_decay,
        eval_strategy="epoch",            # оценка на val после каждой эпохи
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,             # чекпойнты без состояния оптимизатора — экономим диск
        load_best_model_at_end=True,      # в конце возвращаем лучшую по val loss модель
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="epoch",
        report_to="none",
        seed=SEED,
        disable_tqdm=True,
        dataloader_pin_memory=False,      # на mps пининг памяти не нужен
    )

    tracker = RunTracker(run_name)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),  # динамический паддинг по батчу
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience), tracker],
    )

    t_start = time.time()
    trainer.train()
    total_s = time.time() - t_start

    tracker.capture = False  # дальше идут финальные вызовы — историю эпох не трогаем
    val_metrics = trainer.evaluate()  # метрики лучшей (восстановленной) модели на val
    best = min(tracker.history, key=lambda r: r.get("loss", float("inf")))

    summary = {
        "run": run_name,
        "model": model_name,
        "lr": lr,
        "batch_size": batch_size,
        "warmup_ratio": warmup_ratio,
        "params_m": round(model.num_parameters() / 1e6, 1),
        "epochs_trained": len(tracker.history),
        "best_epoch": best["epoch"],
        "epoch_times_s": [r["epoch_time_s"] for r in tracker.history],
        "mean_epoch_time_s": round(float(np.mean([r["epoch_time_s"] for r in tracker.history])), 1),
        "total_train_time_s": round(total_s, 1),
        "val": {k: round(float(v), 4) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "history": tracker.history,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps({"run": run_name, "event": "summary",
                            **{k: v for k, v in summary.items() if k != "history"}},
                           ensure_ascii=False) + "\\n")
    return trainer, summary
"""))

cells.append(code("""\
trainer_base, summary_base = train_model(
    BASE_MODEL, base_ds["train"], base_ds["val"], run_name="rubert-base"
)

display(pd.DataFrame(summary_base["history"]))
print(json.dumps({k: v for k, v in summary_base.items() if k != "history"},
                 ensure_ascii=False, indent=2))
"""))

cells.append(md("ANALYSIS_BASE_SPEED"))

# ============================== 4. оценка качества ==============================
cells.append(md("""\
## 4. Оценка качества на test

Тест — `in_domain_dev.csv`. Считаем все три метрики (accuracy, F1, MCC), чтобы обосновать выбор главной.
"""))

cells.append(code("""\
def predict(trainer, ds):
    \"\"\"Метрики, предсказанные классы и вероятности класса acceptable на датасете.\"\"\"
    out = trainer.predict(ds)
    probs = torch.softmax(torch.tensor(out.predictions), dim=-1)[:, 1].numpy()
    preds = out.predictions.argmax(-1)
    metrics = compute_metrics((out.predictions, out.label_ids))
    return metrics, preds, probs


base_test_metrics, base_preds, base_probs = predict(trainer_base, base_ds["test"])
print("ruBert-base на test (in_domain_dev):",
      {k: round(v, 4) for k, v in base_test_metrics.items()})
"""))

cells.append(md("ANALYSIS_METRIC_CHOICE"))

cells.append(code("""\
def show_errors(df, preds, probs, n=10):
    \"\"\"n случайных ошибок модели: предложение, истинный класс, предсказание, уверенность.\"\"\"
    res = df.assign(pred=preds, p_acceptable=probs.round(3))
    errors = res[res["acceptable"] != res["pred"]]
    print(f"ошибок: {len(errors)} из {len(res)} ({len(errors) / len(res):.1%})")
    return errors.sample(n, random_state=SEED).reset_index(drop=True)


show_errors(test_df, base_preds, base_probs)
"""))

cells.append(md("ANALYSIS_BASE_ERRORS"))

# ============================== 5. rubert-tiny2 ==============================
cells.append(md("""\
## 5. «Маленький BERT»: `cointegrated/rubert-tiny2`

Меняем модель на `cointegrated/rubert-tiny2` (в ~6 раз меньше ruBert-base) и **обязательно заново токенизируем** данные —
у tiny2 собственный словарь. Обучаем по той же процедуре, что и в п. 3.
"""))

cells.append(code("""\
# у tiny2 свой токенизатор/словарь — ретокенизируем данные
tiny_tokenizer, tiny_ds = build_datasets(TINY_MODEL)
print(f"размер словаря: ruBert-base = {len(base_tokenizer)}, rubert-tiny2 = {len(tiny_tokenizer)}")
"""))

cells.append(code("""\
trainer_tiny, summary_tiny = train_model(
    TINY_MODEL, tiny_ds["train"], tiny_ds["val"], run_name="rubert-tiny2"
)

display(pd.DataFrame(summary_tiny["history"]))
print(json.dumps({k: v for k, v in summary_tiny.items() if k != "history"},
                 ensure_ascii=False, indent=2))
"""))

cells.append(md("ANALYSIS_TINY_SPEED"))

# ============================== 6. сравнение ==============================
cells.append(md("""\
## 6. Сравнение моделей

Оцениваем `rubert-tiny2` на test по процедуре из п. 4 и сравниваем с `ruBert-base`:
качество против скорость и размер.
"""))

cells.append(code("""\
tiny_test_metrics, tiny_preds, tiny_probs = predict(trainer_tiny, tiny_ds["test"])
print("rubert-tiny2 на test (in_domain_dev):",
      {k: round(v, 4) for k, v in tiny_test_metrics.items()})


def comparison_row(name, summary, test_metrics):
    return {
        "модель": name,
        "параметры, млн": summary["params_m"],
        "эпох (до плато)": summary["epochs_trained"],
        "сек / эпоху": summary["mean_epoch_time_s"],
        "полное время обучения, сек": summary["total_train_time_s"],
        "val loss": summary["val"]["eval_loss"],
        "val f1": summary["val"]["eval_f1"],
        "test acc": round(test_metrics["acc"], 4),
        "test f1": round(test_metrics["f1"], 4),
        "test mcc": round(test_metrics["mcc"], 4),
    }


comparison = pd.DataFrame([
    comparison_row("ruBert-base", summary_base, base_test_metrics),
    comparison_row("rubert-tiny2", summary_tiny, tiny_test_metrics),
])
comparison
"""))

cells.append(md("ANALYSIS_COMPARISON"))

# ============================== 7. гиперпараметры ==============================
cells.append(md("""\
## 7. Подбор гиперпараметров

Исследуем влияние гиперпараметров на быстрой `rubert-tiny2` (полный grid на ruBert-base занял бы часы):

1. **Grid**: `learning_rate` ∈ {1e-5, 2e-5, 5e-5} × `batch_size` ∈ {8, 16, 32} — 9 запусков;
2. **Warmup**: при лучшей конфигурации грида сравниваем `warmup_ratio` ∈ {0.0, 0.1, 0.3}.

Каждый запуск — до 4 эпох с ранней остановкой по val loss (как в п. 3), метрика сравнения — val F1
(е же требует таблица в задании). Прогресс дублируется в `logs/grid_progress.csv`.
"""))

cells.append(code("""\
grid_rows = []
for lr, bs in itertools.product([1e-5, 2e-5, 5e-5], [8, 16, 32]):
    _, s = train_model(
        TINY_MODEL, tiny_ds["train"], tiny_ds["val"],
        run_name=f"tiny-lr{lr:g}-bs{bs}", lr=lr, batch_size=bs,
    )
    grid_rows.append({
        "lr": lr,
        "batch_size": bs,
        "epochs": s["epochs_trained"],
        "val_f1": round(s["val"]["eval_f1"], 4),
        "val_mcc": round(s["val"]["eval_mcc"], 4),
        "val_loss": round(s["val"]["eval_loss"], 4),
        "сек/эпоха": s["mean_epoch_time_s"],
        "notes": f"лучшая эпоха {s['best_epoch']}",
    })
    pd.DataFrame(grid_rows).to_csv("logs/grid_progress.csv", index=False)  # прогресс на случай перезапуска

grid_df = pd.DataFrame(grid_rows)
grid_df
"""))

cells.append(code("""\
best_row = grid_df.sort_values("val_f1", ascending=False).iloc[0]
print("лучшая конфигурация грида:", dict(best_row[["lr", "batch_size"]]))

warmup_rows = []
for wr in [0.0, 0.1, 0.3]:
    _, s = train_model(
        TINY_MODEL, tiny_ds["train"], tiny_ds["val"],
        run_name=f"tiny-warmup-{wr}", lr=float(best_row["lr"]),
        batch_size=int(best_row["batch_size"]), warmup_ratio=wr,
    )
    warmup_rows.append({
        "warmup_ratio": wr,
        "lr": best_row["lr"],
        "batch_size": best_row["batch_size"],
        "epochs": s["epochs_trained"],
        "val_f1": round(s["val"]["eval_f1"], 4),
        "val_mcc": round(s["val"]["eval_mcc"], 4),
        "notes": f"лучшая эпоха {s['best_epoch']}",
    })

pd.DataFrame(warmup_rows)
"""))

cells.append(md("ANALYSIS_HYPERPARAMS"))

cells.append(md("FINAL_CONCLUSIONS"))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "hw-bert", "language": "python", "name": "hw-bert"},
    "language_info": {"name": "python"},
})

out = ROOT / "hw_bert_rucola.ipynb"
nbf.write(nb, str(out))
print(f"written: {out}")
