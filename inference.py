import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator
import os, numpy as np, torch
from datasets import load_dataset

def validate_model(model, val_set, device):
    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=2,
        pin_memory=True
    )

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1)

            mask = labels != -100
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    accuracy = correct / total
    print(f"Validation accuracy: {accuracy:.4f}")
    assert 0.9 < accuracy < 0.91
    return accuracy

def find_topk_duplicates(query: str, pool: list[str], model, tokenizer, device, top_k: int = 5, batch_size: int = 32):
    pool_clean = [p for p in pool if p != query]
    scores = []
    for i in range(0, len(pool_clean), batch_size):
        chunk = pool_clean[i:i+batch_size]
        enc = tokenizer([query]*len(chunk), chunk,
                        truncation=True, padding=True, max_length=128,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            prob_dup = torch.softmax(logits, dim=-1)[:, 1]
        scores.extend(prob_dup.cpu().numpy().tolist())

    idx = np.argsort(scores)[::-1][:top_k]
    return [(pool_clean[i], float(scores[i])) for i in idx]

def load_pool_questions():
    pool_ds = load_dataset("glue", "qqp", split="train[:1000]")
    q1 = list(pool_ds["question1"])
    q2 = list(pool_ds["question2"])
    pool_questions = q1 + q2
    pool_questions = list({(q or "").strip() for q in pool_questions if isinstance(q, str) and q and q.strip()})
    return pool_questions

# Пример queries (не запускать напрямую)
queries = [
    "How can I be a good programmer?",
    "What is the best way to lose weight fast?",
    "How to learn Python quickly?",
    "Why is the sky blue?",
    "What is the capital of France?",
]