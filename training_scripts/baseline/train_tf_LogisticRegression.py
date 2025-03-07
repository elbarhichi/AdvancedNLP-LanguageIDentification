import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Chargement du CSV contenant les données prétraitées
# On suppose ici que la colonne "Text" contient le texte nettoyé
# et que la colonne "Label" contient le label (code langue)
df = pd.read_csv('data/train_submission_cleaned.csv')

# Vérification rapide de la distribution des classes
print("Distribution des langues dans le dataset:")
print(df['Label'].value_counts())

# Séparation en ensemble d'entraînement et de validation (stratifié sur le label)
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['Label'], random_state=42)

# Création d'un vecteur TF-IDF avec des n-grammes de caractères (par exemple 2 à 4 caractères)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
X_train = vectorizer.fit_transform(train_df['Text'])
X_val = vectorizer.transform(val_df['Text'])

# Entraînement d'un classifieur par régression logistique
clf = LogisticRegression(max_iter=1000, n_jobs=1)
clf.fit(X_train, train_df['Label'].values)

# Prédiction sur l'ensemble de validation
y_pred = clf.predict(X_val)
accuracy = accuracy_score(val_df['Label'], y_pred)
print("Accuracy sur le set de validation:", accuracy)


joblib.dump(clf, 'models/baseline/LogisticRegression/logistic_model.pkl')
joblib.dump(vectorizer, 'models/baseline/LogisticRegression/tfidf_vectorizer.pkl')
