import torch
from dataset import load_qqp_dataset, prepare_datasets
from model import load_model, fine_tune_deberta
from inference import validate_model, load_pool_questions, find_topk_duplicates
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    tokenizer, model = load_model()
    model = model.to(device)

    dataset = load_qqp_dataset()
    qqp_preprocessed, val_set = prepare_datasets(dataset, tokenizer)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False, collate_fn=transformers.default_data_collator
    )
    for batch in val_loader:
        break
    print("Sample batch:", batch)

    with torch.no_grad():
        predicted = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
        )
    print("\nPrediction (probs):", torch.softmax(predicted.logits, dim=1).data.numpy())

    last_ckpt = fine_tune_deberta()

    tokenizer = AutoTokenizer.from_pretrained(last_ckpt, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(last_ckpt).to(device)
    model.eval()

    validate_model(model, val_set, device)

    pool_questions = load_pool_questions()
    queries = [
        "How can I be a good programmer?",
        "What is the best way to lose weight fast?",
        "How to learn Python quickly?",
        "Why is the sky blue?",
        "What is the capital of France?",
    ]
    for q in queries:
        print(f"Query: {q}")
        for cand, s in find_topk_duplicates(q, pool_questions, model, tokenizer, device, top_k=5):
            print(f"{cand} (score={s:.3f})")