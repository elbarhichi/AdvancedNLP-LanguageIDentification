import pandas as pd
import torch
from transformers import (BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Check CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
df = pd.read_csv("data/train_submission.csv")
label_counts = df["Label"].value_counts()
valid_labels = label_counts[label_counts > 10].index
print(f"Number of valid labels: {len(valid_labels)}")
df = df[df["Label"].isin(valid_labels)]

label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])

train_df, test_df = train_test_split(df, test_size=0.000001, random_state=42)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

# Load model and tokenizer
model_ckpt = "google-bert/bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_ckpt)
model = BertForSequenceClassification.from_pretrained(model_ckpt, num_labels=len(valid_labels))
model.to(device)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["Text"], padding="max_length", truncation=True)

# Tokenize
tokenized_dataset = dataset_dict.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.rename_column("Label", "labels")
tokenized_dataset = tokenized_dataset.remove_columns(["Text"])
tokenized_dataset.set_format("torch")

# Training arguments set up
training_args = TrainingArguments(
    output_dir="./results_multilingual_ep5_001",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
)

# Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer
)
trainer.train()


# Evaluate
eval_result = trainer.evaluate()
print("Résultats de l'évaluation :", eval_result)

# Save model and tokenizer
model.save_pretrained("models/bertmulti/bert_multi_classifier")
tokenizer.save_pretrained("models/bertmulti/bert_multi_classifier")


