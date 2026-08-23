# Machine Learning Model Development — Predicting "Luxury" Car Classification

Week 4 internship task: build and evaluate a binary classifier in Python
predicting whether a vehicle is marketed as "Luxury" using only its
objective specifications (no price used as a feature).

## Business Question

Can a vehicle's engine, fuel economy, size, and drivetrain specs predict
whether it's marketed as "Luxury" — useful for auto-tagging incomplete
listings or auditing manufacturer categorization.

## Models Compared

| Model | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 78.4% | 0.772 | 0.676 | 0.721 | 0.832 |
| Decision Tree (depth=6) | 91.4% | 0.885 | 0.910 | 0.898 | 0.969 |

Both models were cross-validated (5-fold) and checked for overfitting via
train/test/CV accuracy comparison and a learning curve analysis.

## Key Finding / Caveat

The Decision Tree's top feature by a wide margin is `popularity` (~60% of
importance), which is a near-constant score per manufacturer in this
dataset — meaning the model may be leaning on brand identity as a proxy
for "luxury" rather than genuine specification signal. See the full report
for discussion and suggested follow-up experiments.

## Files

- `analysis.py` — full pipeline: data prep, preprocessing, model training, evaluation, visualizations
- `cars_clean.csv` — cleaned dataset (from Week 1)
- `results.json` — key metrics for both models
- `viz1_confusion_matrices.png`, `viz2_roc_curves.png`, `viz3_feature_importance.png`, `viz4_learning_curve.png`
- `Week4_ML_Model_Report.docx` — full write-up with methodology and critical discussion

## How to Run

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
python analysis.py
```
