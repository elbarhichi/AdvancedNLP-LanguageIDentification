# Advanced-NLP-LanguageIDentification

This project presents the development of a robust multilingual language classifier. The approach involves thorough exploratory data analysis (EDA), data cleaning, and the evaluation of various models ranging from traditional methods like TF-IDF and FastText to advanced transformer architectures such as BERT, mBERT, DeBERTaV3, mT5, XLM-RoBERTa, and RemBERT. The final model combines these approaches into an ensemble voting system to improve overall classification performance.

## Results

| Model                | Best Validation Score | Best Kaggle Score |
|----------------------|-----------------------|-------------------|
| TF-IDF + SGD         | 0.7536                | 0.7432            |
| FastText             | 0.7820                | 0.7946            |
| mT5                  | 0.8186                | 0.7931            |
| BERT                 | 0.8386                | 0.8267            |
| mDeBERTa             | 0.8451                | 0.8629            |
| mBERT                | 0.8800                | 0.8656            |
| XLM-RoBERTa          | 0.8771                | 0.8710            |
| RemBERT              | 0.8769                | 0.8744            |
| Ensemble Method      | -                     | **0.8950**        |

The **ensemble approach** significantly outperformed individual models, reducing misclassification errors and enhancing language identification performance across various language classes.

## Requirements

- Python 3.7+
- Required libraries:
  - Pandas
  - Scikit-learn
  - FastText
  - HuggingFace Transformers
  - TensorFlow / PyTorch (depending on model choice)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multilingual-language-classifier.git
   cd multilingual-language-classifier
