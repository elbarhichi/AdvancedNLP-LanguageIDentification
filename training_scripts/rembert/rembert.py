import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

df = pd.read_csv('data/train_submission_cleaned.csv')

labels = sorted(df['Label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
df['label_id'] = df['Label'].map(label2id)

# Test / Train split
train_df, val_df = train_test_split(df, test_size=0.02, stratify=df['label_id'], random_state=42)

train_dataset = Dataset.from_pandas(train_df[['Text', 'label_id']].rename(columns={"Text": "text"}))
val_dataset = Dataset.from_pandas(val_df[['Text', 'label_id']].rename(columns={"Text": "text"}))

train_dataset = train_dataset.rename_column("label_id", "labels")
val_dataset = val_dataset.rename_column("label_id", "labels")

# Load tokenizer
model_checkpoint = "google/rembert" 
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load classifier
num_labels = len(labels)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Training set up
training_args = TrainingArguments(
    output_dir="./results_rembert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

# Evaluation
eval_result = trainer.evaluate()
print("Résultats de l'évaluation :", eval_result)

# Saving model
model.save_pretrained("./rembert_language_classifier")
tokenizer.save_pretrained("./rembert_language_classifier")
