import os, re, numpy as np
import glob
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import load_dataset

def load_model(model_name="gchhablani/bert-base-cased-finetuned-qqp"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def fine_tune_deberta():
    dataset = load_dataset("glue", "qqp")
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def preprocess(ex):
        return tokenizer(
            ex["question1"], ex["question2"],
            truncation=True, padding="max_length", max_length=128
        )

    encoded = dataset.map(preprocess, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        labels = labels.astype(int)

        acc = (preds == labels).mean()

        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        return {"accuracy": float(acc), "f1": float(f1)}

    args = TrainingArguments(
        output_dir="./results",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"].shuffle(seed=42).select(range(20000)),
        eval_dataset=encoded["validation"].select(range(2000)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())

    RESULTS_DIR = "./results"
    ckpts = glob(os.path.join(RESULTS_DIR, "checkpoint-*"))
    assert ckpts, "В ./results нет checkpoint-ов. Запусти обучение или проверь путь."

    def step_num(p):
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else -1

    last_ckpt = max(ckpts, key=step_num)
    print("Using checkpoint:", last_ckpt)

    return last_ckpt