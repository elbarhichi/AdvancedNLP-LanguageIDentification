import pandas as pd
import torch
from transformers import pipeline

# 1. Chargement du fichier de test
test_data = pd.read_csv("data/test_without_labels_preprocessed.csv", encoding="utf-8")
test_data = test_data.dropna()
sentences = test_data["Text"].tolist()

# 2. Configuration du modèle et du pipeline
device = 0 if torch.cuda.is_available() else -1
model_ckpt = "models/xlm/xlm-roberta-base-last/checkpoint-17000"
model_pipe = pipeline(
    "text-classification", model=model_ckpt, device=device, batch_size=64
)

# 3. Prédiction des labels
labels = model_pipe(sentences, truncation=True, max_length=128)

# 4. Extraction des résultats et création du DataFrame
y_pred = []
for i in range(len(labels)):
    label = labels[i]["label"]
    y_pred.append({"ID": i + 1, "Label": label})

y_df = pd.DataFrame(y_pred)

# 5. Sauvegarde des prédictions
y_df.to_csv("predictions/predictions_xlmlast2.csv", index=False)
