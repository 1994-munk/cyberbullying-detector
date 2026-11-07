# Cyberbully Detection

A beginner-friendly Python text-classification project using the uploaded
DataTurks cyber-troll dataset format. The code demonstrates data inspection,
cleaning, duplicate and label-conflict audits, a leakage-conscious split, a
TF-IDF/logistic-regression baseline, evaluation, and saving a reusable model.

## 1. Prepare the files

Download `cyberbully_detection.py` and `requirements.txt`. Put your original
`Dataset_for_Detection_of_Cyber-Trolls_1.json` file in the same folder. The
dataset is not included with this starter project.

The data file is JSON Lines: each line is one JSON object with the tweet in
`content` and its label in `annotation.label`. Label `0` means **non
cyber-aggressive**; label `1` means **cyber-aggressive**. This dataset does not
prove that an individual message is bullying.

## 2. Install and run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python cyberbully_detection.py
```

The script prints the raw and cleaned row counts, class counts, a majority-only
comparison, precision/recall/F1 and a confusion matrix. It saves
`cyberbully_baseline.joblib` in the current folder.

## 3. What each stage does

1. **Inspect:** read the newline-delimited records and extract text and labels.
2. **Clean:** decode HTML entities, normalize whitespace, discard empty/unknown
   rows, collapse repeated messages, and remove repeated text with conflicting
   labels. Punctuation is retained because it may carry meaning.
3. **Split:** reserve a stratified 20% test set after duplicate handling so
   repeated identical text cannot appear on both sides.
4. **Train:** convert text into TF-IDF word and two-word phrase features, then
   train class-balanced logistic regression. Compare against a majority-only
   baseline so accuracy is not read in isolation.
5. **Evaluate and save:** inspect per-class precision, recall, F1 and the
   confusion matrix, then save the complete pipeline for later predictions.

## Responsible use

This is a learning baseline, not a bullying detector ready for real-world
moderation. The source labels describe cyber-aggressive versus non-aggressive
text; they do not establish repeated targeting, intent, or harm. Sarcasm,
context, dialect, and dataset imbalance can cause errors. A human should review
context before taking action, and this model should not be used to punish or
profile people.