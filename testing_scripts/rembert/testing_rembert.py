import pandas as pd

# load the test data

test_data = pd.read_csv('data/test_without_labels_preprocessed.csv')

import re
import unicodedata

def clean_text(text):
    # Supprimer les URLs
    text = re.sub(r'http\S+', '', text)
    # Supprimer les mentions (@) et hashtags (#)
    text = re.sub(r'[@#]\w+', '', text)
    # Normaliser Unicode (NFC)
    text = unicodedata.normalize('NFC', text)
    # Conversion en minuscules
    text = text.lower()
    # Supprimer la ponctuation si nécessaire (attention à conserver les caractères spécifiques)
    text = re.sub(r'[^\w\s]', '', text)
    # Supprimer les espaces superflus
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Appliquer le nettoyage sur la colonne contenant le texte (par exemple 'text')
test_data['Text'] = test_data['Text'].apply(clean_text)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Chemin vers le modèle sauvegardé
model_path = "models/rembert/rembert_language_classifier"

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Créer un pipeline de classification de texte
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, max_length=512)


# Appliquer le pipeline sur tous les textes en batch
texts = test_data['Text'].tolist()
predictions = classifier(texts, batch_size=16)

# Extraire le label prédit et l'ajouter dans une nouvelle colonne 'predicted_label'
labels = [pred['label'] for pred in predictions]

# Afficher quelques résultats
print(labels[:5])


# save the predictions to a csv file with ID name and Label name id from 1 to len(y_pred) and Label name as y_pred
df = pd.DataFrame({'ID': range(1, len(labels) + 1), 'Label': labels})
df.to_csv('predictions/predictions_rembert.csv', index=False)


