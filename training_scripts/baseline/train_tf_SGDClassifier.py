import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/train_submission_cleaned.csv')

# Quick check of the class distributions
print("Distribution des langues dans le dataset:")
print(df['Label'].value_counts())

# Train / Test split
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['Label'], random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
X_train = vectorizer.fit_transform(train_df['Text'])
X_val = vectorizer.transform(val_df['Text'])

# Linear classifier that usesStochastic Gradient Descent (SGD)
clf = SGDClassifier(loss='log_loss', max_iter=1000, n_jobs=-1)
clf.fit(X_train, train_df['Label'])


# Prediction on validation set
y_pred = clf.predict(X_val)
accuracy = accuracy_score(val_df['Label'], y_pred)
print("Accuracy sur le set de validation:", accuracy)


joblib.dump(clf, 'models/baseline/SGDClassifier/new_logistic_model.pkl')
joblib.dump(vectorizer, 'models/baseline/SGDClassifier/new_tfidf_vectorizer.pkl')
