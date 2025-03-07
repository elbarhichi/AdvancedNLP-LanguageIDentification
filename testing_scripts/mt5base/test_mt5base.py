import pandas as pd
import re
import unicodedata
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline

# 1. Charger le fichier de test
test_data = pd.read_csv('data/test_without_labels_preprocessed.csv')

# 2. Définir une fonction de nettoyage
def clean_text(text):
    # Supprimer les URLs
    text = re.sub(r'http\S+', '', text)
    # Supprimer les mentions (@) et hashtags (#)
    text = re.sub(r'[@#]\w+', '', text)
    # Normaliser Unicode (NFC)
    text = unicodedata.normalize('NFC', text)
    # Conversion en minuscules
    text = text.lower()
    # Supprimer la ponctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Supprimer les espaces superflus
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Appliquer le nettoyage sur la colonne 'Text'
test_data['Text'] = test_data['Text'].apply(clean_text)

# 3. Ajouter le prompt pour mT5-base
test_data['input_text'] = "identify language: " + test_data['Text']

# 4. Charger le tokenizer et le modèle fine-tuné
model_path = "./mt5_base_language_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# 5. Créer un pipeline de génération text-to-text
# Ici, on fixe max_length à 10, car les labels sont courts (ex : "en", "fr", etc.)
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=10,
    device=0  # utiliser le GPU si disponible
)

# 6. Générer les prédictions pour chaque texte
inputs = test_data['input_text'].tolist()
predictions = generator(inputs, batch_size=8)

# Extraire les prédictions
predicted_labels = [pred['generated_text'].strip() for pred in predictions]

# 7. Créer le DataFrame de soumission et sauvegarder en CSV
submission = pd.DataFrame({
    'ID': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('predictions/predictions_mt5_base.csv', index=False)
print(submission.head())
