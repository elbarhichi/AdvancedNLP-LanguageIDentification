import pandas as pd
import fasttext
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import itertools

# 1. Charger le jeu d'entraînement
df = pd.read_csv('data/train_submission_cleaned.csv')
# S'assurer que les colonnes sont bien des chaînes de caractères
df['Text'] = df['Text'].astype(str)
df['Label'] = df['Label'].astype(str)

# 2. Fonction pour convertir le DataFrame en format fastText
def create_fasttext_file(df, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            label = row['Label'].strip()
            text = row['Text'].strip().replace('\n', ' ')
            # Format fastText : __label__<label> <text>
            f.write(f"__label__{label} {text}\n")

# 3. Définir la grille d'hyperparamètres à tester
param_grid = {
    'lr': [0.3, 0.5],
    'epoch': [25, 50, 75],
    'wordNgrams': [1, 2],
    'dim': [100]
}

# 4. Créer un objet StratifiedKFold pour une validation croisée stratifiée à 3 plis (par exemple)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Nombre d'exemples dans le dataset:", len(df))

# 5. Fonction d'évaluation pour une configuration donnée
def evaluate_params(params, df):
    accuracies = []
    fold = 1
    # Utiliser la colonne 'Label' pour la stratification
    for train_index, val_index in skf.split(df, df['Label']):
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        # Créer des noms de fichiers uniques en incorporant les paramètres et le numéro de pli
        train_file = f"temp_train_fold{fold}_lr{params['lr']}_epoch{params['epoch']}_w{params['wordNgrams']}_dim{params['dim']}.txt"
        val_file = f"temp_val_fold{fold}_lr{params['lr']}_epoch{params['epoch']}_w{params['wordNgrams']}_dim{params['dim']}.txt"
        create_fasttext_file(train_df, train_file)
        create_fasttext_file(val_df, val_file)
        print(f"Evaluating fold {fold} with parameters {params} using files:\n  {train_file}\n  {val_file}")
        # Entraîner le modèle avec la configuration actuelle
        model = fasttext.train_supervised(
            input=train_file,
            lr=params['lr'],
            epoch=params['epoch'],
            wordNgrams=params['wordNgrams'],
            dim=params['dim'],
            verbose=2,
            loss='softmax'
        )
        print("Model trained.")
        # Tester sur le fichier de validation (resultat: (nb_exemples, precision, recall))
        result = model.test(val_file)
        accuracy = result[1]  # On considère la précision comme l'accuracy
        print(f"Fold {fold} accuracy: {accuracy}")
        accuracies.append(accuracy)
        
        # Note : On ne supprime pas les fichiers, ils restent pour vérification.
        fold += 1
    return np.mean(accuracies)

# 6. Recherche par grille
best_params = None
best_accuracy = 0
results = []

for lr, epoch, wordNgrams, dim in itertools.product(param_grid['lr'],
                                                      param_grid['epoch'],
                                                      param_grid['wordNgrams'],
                                                      param_grid['dim']):
    params = {'lr': lr, 'epoch': epoch, 'wordNgrams': wordNgrams, 'dim': dim}
    print("Testing parameters:", params)
    avg_acc = evaluate_params(params, df)
    print("Average accuracy:", avg_acc)
    results.append((params, avg_acc))
    if avg_acc > best_accuracy:
        best_accuracy = avg_acc
        best_params = params

print("Best parameters found:", best_params, "with accuracy:", best_accuracy)

# 7. Entraîner le modèle final sur l'ensemble complet avec la meilleure configuration
train_file_full = "fasttext_train_full.txt"
create_fasttext_file(df, train_file_full)

final_model = fasttext.train_supervised(
    input=train_file_full,
    lr=best_params['lr'],
    epoch=best_params['epoch'],
    wordNgrams=best_params['wordNgrams'],
    dim=best_params['dim'],
    verbose=2,
    loss='softmax'
)

# 8. Sauvegarder le modèle final
final_model.save_model("fasttext_language_classifier_best.bin")
print("Final model saved with best parameters.")
