<div align="center">
  <img src="assets/cyberbullying_demo.gif" alt="Colorful animated scene showing a shield mascot flagging hurtful messages and encouraging kindness online" width="980" />
</div>

# 🛡️ Cyberbullying Detector

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-F37726?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img alt="Scikit-learn" src="https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
</p>

This project uses machine learning to detect harmful and aggressive language in social media comments. It is built as a practical Python + Jupyter workflow that takes a raw dataset, cleans the text, trains a text classifier, and evaluates how well it can flag cyberbullying patterns.

The purpose is educational and research-oriented: the system helps identify risky language for human review, but it is not a final judgment on a person or a substitute for common sense and context.

## 🌈 Why this project matters

Cyberbullying can damage confidence, safety, and mental well-being. By training a model on real examples of aggressive language, this project shows how AI can help communities flag harmful content earlier and support safer online spaces.

## ✨ What the project does

- Loads tweet data from a JSON-lines dataset
- Extracts text and labels from the raw records
- Cleans text by normalizing whitespace and HTML content
- Removes invalid or conflicting labels before training
- Splits the dataset into train/test sets
- Trains a TF-IDF + logistic regression model
- Evaluates performance with precision, recall, F1-score, and confusion matrix
- Saves the trained model for later use
- Includes a notebook with clear, commented code for learning

## 🧠 Model approach

This project uses a simple but effective supervised learning workflow:

- TF-IDF converts text into numerical features
- Logistic regression learns the pattern between aggressive and non-aggressive comments
- Class balancing helps the model better handle skewed label distributions
- The model is easy to understand and easy to improve later

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
├── data/
│   ├── README.md
│   └── Dataset_for_Detection_of_Cyber-Trolls_1.json
└── cyberbully_baseline.joblib
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

### 3) Open the notebook

```bash
jupyter notebook notebooks/cyberbullying_detector.ipynb
```

### 4) Train the model from the terminal

```bash
python train_model.py --dataset data/Dataset_for_Detection_of_Cyber-Trolls_1.json --output cyberbully_baseline.joblib
```

## 📓 Notebook workflow

The notebook includes clear sections for:

1. Loading and inspecting the raw dataset
2. Extracting content and labels
3. Cleaning text and filtering bad rows
4. Removing conflicting duplicate entries
5. Splitting into train/test data
6. Training the classifier
7. Evaluating model results
8. Saving the model for future predictions

## ⚠️ Important note

This project is designed to detect harmful and aggressive language as a signal for review, not to prove that a person is a bully. Human context, tone, and intent are crucial. The model should support moderation decisions, not replace them.

## 🧪 Sample output

```text
              precision    recall  f1-score   support

  non-aggressive       0.90      0.77      0.83      2358
cyber-aggressive       0.39      0.63      0.48       552

        accuracy                           0.74      2910
       macro avg       0.64      0.70      0.65      2910
    weighted avg       0.80      0.74      0.76      2910
```

## 🧾 License

This project is intended for educational and research use.

---

<p align="center">
  <strong>Built for safer online communities with AI for early detection and human review.</strong>
</p>
