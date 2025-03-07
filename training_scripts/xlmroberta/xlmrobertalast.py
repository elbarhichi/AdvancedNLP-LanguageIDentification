import time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)


train_data = pd.read_csv("data/train_submission_preprocessed.csv", encoding="utf-8")
train_data = train_data.dropna()

# Test / Train split
sentences, labels = train_data["Text"], train_data["Label"]
sentences_train, sentences_test, labels_train, labels_test = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)
sentences_val, sentences_test, labels_val, labels_test = train_test_split(
    sentences_test, labels_test, test_size=0.5, random_state=42
)

ds_train = Dataset.from_dict({"text": sentences_train, "labels": labels_train})
ds_valid = Dataset.from_dict({"text": sentences_val, "labels": labels_val})
ds_test = Dataset.from_dict({"text": sentences_test, "labels": labels_test})

print(f"Train / valid / test samples: {len(ds_train)} / {len(ds_valid)} / {len(ds_test)}")

# Load tokenizer XLM-Roberta (base)
model_ckpt = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_text(sequence):
    return tokenizer(sequence["text"], truncation=True, max_length=128)

def encode_labels(example):
    example["labels"] = label2id[example["labels"]]
    return example

tok_train = ds_train.map(tokenize_text, batched=True)
tok_valid = ds_valid.map(tokenize_text, batched=True)
tok_test = ds_test.map(tokenize_text, batched=True)


all_langs = np.unique(labels)
id2label = {idx: all_langs[idx] for idx in range(len(all_langs))}
label2id = {v: k for k, v in id2label.items()}

tok_train = tok_train.map(encode_labels, batched=False)
tok_valid = tok_valid.map(encode_labels, batched=False)
tok_test = tok_test.map(encode_labels, batched=False)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load pretrained classifier
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=len(all_langs), id2label=id2label, label2id=label2id
)

# Accuracy
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Training set up
train_bs = 64
eval_bs = train_bs * 2
logging_steps = len(tok_train) // train_bs
output_dir = "models/xlm-roberta-base-last"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=train_bs,
    per_device_eval_batch_size=eval_bs,
    evaluation_strategy="epoch",
    logging_steps=logging_steps,
    fp16=True,
)

# Training
trainer = Trainer(
    model,
    training_args,
    compute_metrics=compute_metrics,
    train_dataset=tok_train,
    eval_dataset=tok_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()

# Evaluation
device = 0 if torch.cuda.is_available() else -1
model_ckpt = "models/xlm-roberta-base-last"
pipe = pipeline("text-classification", model=model_ckpt, device=device)

start_time = time.perf_counter()
model_preds = [
    s["label"]
    for s in pipe(ds_test.text.values.tolist(), truncation=True, max_length=128)
]
print(f"{time.perf_counter() - start_time:.2f} seconds")
print(classification_report(ds_test.labels.values.tolist(), model_preds, digits=3))
