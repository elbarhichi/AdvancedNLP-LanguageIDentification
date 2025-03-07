import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# 1. Chargement du CSV avec les données prétraitées
df = pd.read_csv('data/train_submission_cleaned.csv')

# 2. Remappage des labels en entiers
labels = sorted(df['Label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
df['label_id'] = df['Label'].map(label2id)

# 3. Séparation en ensembles d'entraînement et de validation (stratifiée)
train_df, val_df = train_test_split(df, test_size=0.02, stratify=df['label_id'], random_state=42)

# 4. Création des datasets Hugging Face à partir des DataFrames pandas
train_dataset = Dataset.from_pandas(train_df[['Text', 'label_id']].rename(columns={"Text": "text"}))
val_dataset = Dataset.from_pandas(val_df[['Text', 'label_id']].rename(columns={"Text": "text"}))

# Renommer la colonne "label_id" en "labels" pour le Trainer
train_dataset = train_dataset.rename_column("label_id", "labels")
val_dataset = val_dataset.rename_column("label_id", "labels")

# 5. Chargement du tokenizer et du modèle mDeBERTa-v3-base
model_checkpoint = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# 6. Définir une fonction de tokenization
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

# Appliquer la tokenization sur les datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Spécifier les colonnes à utiliser pour l'entraînement (format PyTorch)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 7. Charger le modèle pré-entraîné pour la classification
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
    output_dir="./results_mdeberta_base",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # Vous pouvez ajuster la batch size en fonction de votre GPU
    per_device_eval_batch_size=16,
    num_train_epochs=9,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # Activez fp16 si votre GPU le supporte
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
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

# 13. Sauvegarder le modèle et le tokenizer pour une utilisation ultérieure
model.save_pretrained("models/mdebertav3base/mdeberta_v3_base_language_classifier")
tokenizer.save_pretrained("models/mdebertav3base/mdeberta_v3_base_language_classifier")
