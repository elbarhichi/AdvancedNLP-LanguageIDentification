# Advanced-NLP-LanguageIDentification

This repository contains code, scripts, and notebooks developed by Group 5 for the NLP CS Kaggle Challenge on multilingual language identification, involving the classification of text snippets into 389 distinct language labels.

## Project Overview
We aimed to build a robust system that identifies the language of a given text snippet among 389 classes. Our approach includes:
- Thorough data cleaning (handling duplicates, mislabeled samples, removing noise).
- Baseline methods (TF-IDF + linear models, fastText).
- Advanced LLMs (BERT, mBERT, DeBERTa-v3, XLM-RoBERTa, RemBERT, mT5).
- An ensemble voting strategy to combine model outputs.

## Dataset

The dataset was provided as part of a Kaggle challenge. It includes approximately 190,100 cleaned samples across 389 languages after rigorous preprocessing (removal of mislabeled samples, duplicates, noise, normalization).

## Structure and Content

- **Preprocessing notebooks:**
  - `1_preprocessing.ipynb`: Initial EDA and data anomaly detection.
  - `2_cleaning.ipynb`: Cleaning pipeline (duplicates, mislabeled, noise removal).
  - `3_final_evaluation.ipynb`: Producing final predictions based on Ensemble Voting

- **Baseline Models**: `baseline` directory (TF-IDF + Logistic Regression / SGD Classifier).
- **FastText**: `fasttext` directory contains scripts for grid search and hyperparameter tuning.
- **Advanced Transformers**: Separate directories (`bert`, `bertmulti`, `mdebertav3`, `mt5base`, `xlmroberta`, `rembert`) for each advanced model.
- **Ensemble Strategy**: Implemented in the notebook `3_final_evaluation.ipynb`.

## Usage

To reproduce our results:

1. Clone this repository:
   ```bash
   git clone https://github.com/elbarhichi/AdvancedNLP-LanguageIDentification.git
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Navigate to the desired model's directory in `training_scripts` or `testing_scripts`.


4. Follow detailed instructions provided within individual scripts or notebooks.

## Results

| Model                  | Validation Accuracy | Kaggle Score |
|------------------------|---------------------|--------------|
| TF-IDF + SGD           | 0.7536              | 0.7432       |
| FastText               | 0.7820              | 0.7946       |
| mT5                    | 0.8186              | 0.7931       |
| BERT                   | 0.8386              | 0.8267       |
| mDeBERTa               | 0.8451              | 0.8629       |
| mBERT                  | 0.8800              | 0.8656       |
| XLM-RoBERTa            | 0.8771              | 0.8710       |
| RemBERT                | 0.8769              | 0.8745       |
| **Ensemble Method**    | **-**               | **0.8950**   |



## Contributors

- Malek Bouhadida
- Mohammed El Barhichi
- Abdelaziz Guelfane
- Imane Meziany
- Yousra Yakhou
