import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# 1. Chargement du CSV
df = pd.read_csv('data/train_submission_cleaned.csv')

# 2. Remapper les labels en entiers
labels = sorted(df['Label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
df['label_id'] = df['Label'].map(label2id)

# 3. Séparation en train/validation (stratifiée)
train_df, val_df = train_test_split(df, test_size=0.05, stratify=df['label_id'], random_state=42)

# 4. Création de datasets Hugging Face à partir de Pandas
train_dataset = Dataset.from_pandas(train_df[['Text', 'label_id']].rename(columns={"Text": "text"}))
val_dataset = Dataset.from_pandas(val_df[['Text', 'label_id']].rename(columns={"Text": "text"}))

# Renommer la colonne "label_id" en "labels"
train_dataset = train_dataset.rename_column("label_id", "labels")
val_dataset = val_dataset.rename_column("label_id", "labels")

# 5. Charger le tokenizer et le modèle RemBERT
model_checkpoint = "google/rembert"  # Vous pouvez tester aussi "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 6. Fonction de tokenization
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 7. Charger le modèle pour la classification
num_labels = len(labels)
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# 8. Définir la fonction de calcul de l'accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 9. Configurer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results_rembert_overfit",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,              # Learning rate un peu plus élevé
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=14,             # Augmenter le nombre d'époques
    warmup_steps=500,
    weight_decay=0.001,                # Désactiver ou réduire la régularisation
    fp16=True,
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=13,
    load_best_model_at_end=False,    # Optionnel: chargez le dernier modèle, car overfitting est attendu
    metric_for_best_model="accuracy"
)


# 10. Créer l'objet Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 11. Lancer l'entraînement
trainer.train()

# 12. Évaluer le modèle sur le set de validation
eval_result = trainer.evaluate()
print("Résultats de l'évaluation :", eval_result)

# 13. Sauvegarder le modèle et le tokenizer
model.save_pretrained("models/rembert/rembert_language_classifier2")
tokenizer.save_pretrained("models/rembert/rembert_language_classifier2")
