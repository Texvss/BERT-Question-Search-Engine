import datasets
from transformers import AutoTokenizer

def load_qqp_dataset():
    return datasets.load_dataset("SetFit/qqp")

def preprocess_function(examples, tokenizer, max_length=128):
    result = tokenizer(
        examples["text1"],
        examples["text2"],
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    result["label"] = examples["label"]
    return result

def prepare_datasets(dataset, tokenizer, max_length=128):
    qqp_preprocessed = dataset.map(lambda ex: preprocess_function(ex, tokenizer, max_length), batched=True)
    val_set = qqp_preprocessed["validation"]
    return qqp_preprocessed, val_set