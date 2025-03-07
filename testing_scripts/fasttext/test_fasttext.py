import pandas as pd
import fasttext
import re
import unicodedata

# 1. Charger la data de test
test_data = pd.read_csv('data/test_without_labels_preprocessed.csv')

# 2. Définir une fonction de nettoyage (comme précédemment)
def clean_text(text):
    # Supprimer les URLs
    text = re.sub(r'http\S+', '', text)
    # Supprimer les mentions (@) et hashtags (#)
    text = re.sub(r'[@#]\w+', '', text)
    # Normaliser Unicode (NFC)
    text = unicodedata.normalize('NFC', text)
    # Conversion en minuscules
    text = text.lower()
    # Supprimer la ponctuation (attention à conserver les caractères utiles)
    text = re.sub(r'[^\w\s]', '', text)
    # Supprimer les espaces superflus
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Appliquer le nettoyage sur la colonne 'Text'
test_data['Text'] = test_data['Text'].apply(clean_text)

# 3. Charger le modèle fastText sauvegardé
model = fasttext.load_model("fasttext_language_classifier_best.bin")

# 4. Faire des prédictions pour chaque texte
predicted_labels = []
for text in test_data['Text']:
    # La méthode predict renvoie une liste de labels et leurs probabilités
    labels, probs = model.predict(text)
    # fastText retourne par défaut des labels sous forme "__label__xxx"
    # On retire le préfixe pour ne garder que "xxx"
    pred = labels[0].replace("__label__", "")
    predicted_labels.append(pred)

# 5. Créer un DataFrame pour la soumission (ID de 1 à n)
submission = pd.DataFrame({'ID': range(1, len(predicted_labels) + 1),
                           'Label': predicted_labels})

# 6. Sauvegarder le DataFrame en CSV
submission.to_csv('predictions/predictions_fasttext2.csv', index=False)

# Afficher quelques lignes pour vérification
print(submission.head())
