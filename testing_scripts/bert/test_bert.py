
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import pandas as pd
from transformers import (BertTokenizer, BertForSequenceClassification)
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Chargement des données de test
input_csv = "data/test_without_labels.csv"
df_test = pd.read_csv(input_csv)

# 2. Chargement du modèle et du tokenizer
model_path = "./results_bert"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
label_encoder = LabelEncoder()

# 2. Tokenization des textes
test_encoded = tokenizer(df_test["Text"].tolist(), padding=True, truncation=True, return_tensors="pt")
input_ids = test_encoded["input_ids"]
attention_mask = test_encoded["attention_mask"]

dataset = TensorDataset(input_ids, attention_mask)
dataloader = DataLoader(dataset, batch_size=16)

# 3. Prédiction
test_predictions = []
model.eval()

with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        batch_preds = torch.argmax(outputs.logits, dim=-1).tolist()
        test_predictions.extend(batch_preds)

# 4. Conversion des prédictions en labels réels
real_labels = label_encoder.inverse_transform(test_predictions)
df_results = pd.DataFrame({"ID": range(1, len(real_labels) + 1), "Label": real_labels})
df_results.to_csv("predictions/predictions_bert.csv", index=False, encoding="utf-8")

print("Predictions saved with original labels!")