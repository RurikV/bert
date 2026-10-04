"""Быстрая проверка API перед полным запуском: tiny2, 160 примеров, 1 эпоха.

Проверяет: TrainingArguments (eval_strategy/save_only_model/...), processing_class,
EarlyStoppingCallback + кастомный колбэк, Trainer на mps, predict.
"""
import time

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import matthews_corrcoef
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
torch.manual_seed(SEED)

df = pd.read_csv("data/in_domain_train.csv")[["sentence", "acceptable"]]
tr, va = train_test_split(df, test_size=0.2, stratify=df["acceptable"], random_state=SEED)
tr, va = tr.head(160), va.head(40)

tok = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
mk = lambda d: (
    Dataset.from_pandas(d, preserve_index=False)
    .rename_column("acceptable", "label")
    .map(lambda b: tok(b["sentence"], truncation=True, max_length=64),
         batched=True, remove_columns=["sentence"])
)
trs, vas = mk(tr), mk(va)

model = AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny2", num_labels=2)
args = TrainingArguments(
    output_dir="outputs/smoke",
    num_train_epochs=1,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    save_only_model=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_strategy="epoch",
    report_to="none",
    seed=SEED,
    disable_tqdm=True,
    dataloader_pin_memory=False,
)


def cm(p):
    lg, y = p
    pr = lg.argmax(-1)
    return {"acc": float((pr == y).mean()), "mcc": float(matthews_corrcoef(y, pr))}


class T(TrainerCallback):
    def on_epoch_begin(self, a, s, c, **k):
        self.t0 = time.time()

    def on_epoch_end(self, a, s, c, metrics=None, **k):
        print("EPOCH_END:", {k: v for k, v in (metrics or {}).items()})


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trs,
    eval_dataset=vas,
    processing_class=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=cm,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1), T()],
)
trainer.train()

out = trainer.predict(vas)
print("predict ok:", out.predictions.shape, out.metrics)
print("num_parameters:", model.num_parameters())
print("SMOKE OK")
