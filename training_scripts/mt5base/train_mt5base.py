import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

# 1. Charger le CSV d'entraînement
df = pd.read_csv('data/train_submission_cleaned.csv')
df['Text'] = df['Text'].astype(str)
df['Label'] = df['Label'].astype(str)

# 2. Créer le prompt pour chaque texte
df['input_text'] = "identify language: " + df['Text']
df['target_text'] = df['Label']

# 3. Séparation en train et validation (stratifiée par Label)
train_df, val_df = train_test_split(df, test_size=0.05, stratify=df['Label'], random_state=42)

# 4. Convertir en datasets Hugging Face
train_dataset = Dataset.from_pandas(train_df[['input_text', 'target_text']])
val_dataset = Dataset.from_pandas(val_df[['input_text', 'target_text']])

# 5. Charger le tokenizer et le modèle mT5‑base
model_checkpoint = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# 6. Prétraitement : tokenisation des entrées et cibles
max_source_length = 256  # ajustez selon la longueur de vos textes
max_target_length = 10   # les labels sont courts

def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Mettre en format PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 7. Configurer les arguments d'entraînement
training_args = Seq2SeqTrainingArguments(
    output_dir="./results_mt5_base",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 8. Fonction de calcul des métriques (corrigée pour éviter l'OverflowError)
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Itérer sur chaque prédiction en convertissant le tableau numpy en liste d'entiers
    decoded_preds = [tokenizer.decode(pred.tolist(), skip_special_tokens=True) for pred in preds]
    # Pour les labels, remplacer -100 par le pad token et convertir chaque exemple en liste
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = [tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in labels]
    # Nettoyer les espaces
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    correct = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l)
    accuracy = correct / len(decoded_preds)
    return {"accuracy": accuracy}

# 9. Créer le Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 10. Lancer l'entraînement
trainer.train()

# 11. Évaluer sur le set de validation
eval_result = trainer.evaluate()
print("Evaluation results:", eval_result)

# 12. Sauvegarder le modèle et le tokenizer
model.save_pretrained("./mt5_base_language_classifier")
tokenizer.save_pretrained("./mt5_base_language_classifier")
