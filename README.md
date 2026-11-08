<div align="center">
  <img src="assets/cyberbullying_demo.gif" alt="Cyberbullying detector demo animation" width="980" />
</div>

# 🛡️ Cyberbullying Detector

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-F37726?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img alt="Scikit-learn" src="https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
</p>

This project builds a practical machine-learning workflow for spotting cyber-aggressive language in social media text. It walks through the full pipeline: loading the dataset, cleaning the comments, fixing label issues, splitting data safely, training a text model, and evaluating the results.

It is designed as a learning project and a research starter, not as an automated moderation verdict system. The model is a helpful signal for review, not a replacement for human judgment.

## ✨ What this project does

- Loads a JSON-lines dataset of social media messages and labels.
- Cleans text by normalizing whitespace and handling missing values.
- Filters invalid and conflicting labels before training.
- Splits the data so the evaluation stays fair.
- Trains a TF-IDF + logistic regression classifier.
- Saves the trained pipeline for future predictions.
- Includes a Jupyter notebook to walk through the workflow step by step.

## 🎨 Why this matters

Cyberbullying is a harmful pattern of online aggression that can escalate quickly and affect mental health, community trust, and safety. This project gives a transparent, beginner-friendly entry point into how text classifiers can help detect harmful language while keeping human oversight in the loop.

## 🧠 Model approach

The project uses a classic and explainable baseline:

- TF-IDF vectorizer for text feature extraction
- Logistic regression classifier for fast, interpretable predictions
- Class-balanced training for better handling of skewed labels
- Evaluation metrics including precision, recall, F1-score, and confusion matrix

This is a strong baseline for education and experimentation, and it is easy to extend with more advanced models later.

## 📁 Project structure

```text
cyberbullying-detector/
├── README.md
├── pyproject.toml
├── requirements.txt
├── train_model.py
├── cyberbullying_detector/
│   ├── __init__.py
│   └── model.py
├── notebooks/
│   └── cyberbullying_detector.ipynb
├── assets/
│   └── cyberbullying_demo.gif
└── data/
    └── Dataset_for_Detection_of_Cyber-Trolls_1.json
```

## 🚀 Quick start

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the notebook

```bash
jupyter notebook notebooks/cyberbullying_detector.ipynb
```

### 4) Train the model from the terminal

```bash
python train_model.py --dataset data/Dataset_for_Detection_of_Cyber-Trolls_1.json --output cyberbully_baseline.joblib
```

## 🔎 Notebook workflow

The notebook covers the full process:

1. Data inspection and label review
2. Text cleaning and normalization
3. Duplicate and conflicting-label handling
4. Train/test splitting
5. Model training
6. Evaluation and model export

## ⚠️ Important note

This repository focuses on detecting cyber-aggressive language as a signal for review. It does not prove that a person is bullying someone, and it should never be used as the sole basis for punishment or public shaming.

## 🧪 Example output

```text
              precision    recall  f1-score   support

           0       0.90      0.91      0.90      1600
           1       0.72      0.69      0.70       650

    accuracy                           0.86      2250
    macro avg       0.81      0.80      0.80      2250
weighted avg       0.86      0.86      0.86      2250
```

## 🧾 License

This project is for educational and research use.

---

<p align="center">
  <strong>Built for safer online communities through transparent AI and human review.</strong>
</p>
