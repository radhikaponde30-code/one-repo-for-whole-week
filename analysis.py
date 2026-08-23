"""
Week 4 Task - Machine Learning Model Development and Evaluation
Dataset: Car Features and MSRP (cleaned in Week 1)

Business problem: Predict whether a vehicle listing is marketed as
"Luxury" (derived from the Market Category field) using only its
objective specifications (price, engine specs, fuel economy, size) -
i.e. can we tell if a car is a luxury model just from its numbers,
without reading the marketing copy? This has real value for tasks like
auto-tagging incomplete listings or auditing manual categorization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                              accuracy_score, precision_score, recall_score, f1_score,
                              RocCurveDisplay)

sns.set_theme(style="whitegrid")
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 150
RANDOM_STATE = 42

# -----------------------------------------------------------------------
# 1. DATA PREPARATION
# -----------------------------------------------------------------------
df = pd.read_csv('cars_clean.csv')

# Target: binary label, 1 if "Luxury" appears in Market Category, else 0.
# Vehicles with no category info ("Not Specified") are excluded, since we
# cannot know their true label - including them would introduce label noise.
df = df[df['market_category'] != 'Not Specified'].copy()
df['is_luxury'] = df['market_category'].str.contains('Luxury').astype(int)

print("Dataset after excluding unlabeled rows:", df.shape)
print("Class balance:\n", df['is_luxury'].value_counts(normalize=True))

# Feature selection: only objective, non-leaking specifications are used.
# MSRP is deliberately EXCLUDED as a feature even though it's predictive -
# using price to predict "luxury" would be circular for many real use
# cases (e.g. auto-tagging a used-market listing where price is what
# we're often also trying to explain). Engine/fuel/size specs are kept
# since they describe the vehicle itself, not its market label.
numeric_features = ['engine_hp', 'engine_cylinders', 'number_of_doors',
                     'highway_mpg', 'city_mpg', 'popularity', 'year']
categorical_features = ['vehicle_size', 'driven_wheels', 'transmission_type']

X = df[numeric_features + categorical_features]
y = df['is_luxury']

# Drop any remaining rows with missing values in the selected features
# (should be none after Week 1 cleaning, verified here for safety)
mask = X.notna().all(axis=1)
X, y = X[mask], y[mask]
print(f"\nFinal modeling set: {X.shape[0]} rows, {X.shape[1]} raw features")

# 80/20 train-test split, stratified to preserve class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

# Preprocessing pipeline: scale numeric features (required for logistic
# regression to converge properly and treat features fairly), one-hot
# encode categoricals. Built as a single ColumnTransformer so it can be
# fit on training data only and applied identically to the test set,
# avoiding data leakage.
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
])

# -----------------------------------------------------------------------
# 2. MODEL SELECTION AND TRAINING
# -----------------------------------------------------------------------
# Two simple, interpretable algorithms are compared:
# - Logistic Regression: a strong, well-calibrated linear baseline, easy
#   to interpret via coefficients, appropriate given the roughly linear
#   separability expected between luxury/non-luxury specs.
# - Decision Tree: captures non-linear interactions (e.g. "high HP AND
#   Rear-wheel drive") that a linear model would miss, and is equally
#   interpretable via its rules. Max depth is limited to control
#   overfitting given the tree's tendency to memorize training data.

log_reg = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
log_reg.fit(X_train, y_train)

tree = Pipeline([
    ('prep', preprocessor),
    ('clf', DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE)),
])
tree.fit(X_train, y_train)

models = {'Logistic Regression': log_reg, 'Decision Tree': tree}

# -----------------------------------------------------------------------
# 3. EVALUATION
# -----------------------------------------------------------------------
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # 5-fold cross-validation on the training set, to check stability of
    # accuracy beyond a single train/test split
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    train_acc = accuracy_score(y_train, model.predict(X_train))

    results[name] = dict(y_pred=y_pred, y_proba=y_proba, acc=acc, prec=prec, rec=rec, f1=f1,
                          fpr=fpr, tpr=tpr, roc_auc=roc_auc, cv_mean=cv_scores.mean(),
                          cv_std=cv_scores.std(), train_acc=train_acc)
    print(f"\n=== {name} ===")
    print(f"Train accuracy: {train_acc:.3f} | Test accuracy: {acc:.3f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Non-Luxury', 'Luxury']))

# -----------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------

# Viz 1: Confusion matrices, side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Non-Luxury', 'Luxury'], yticklabels=['Non-Luxury', 'Luxury'])
    ax.set_title(f"{name}\nAccuracy = {res['acc']:.2%}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig('viz1_confusion_matrices.png', dpi=150)
plt.close()

# Viz 2: ROC curves, both models overlaid
fig, ax = plt.subplots(figsize=(7, 6))
colors = {'Logistic Regression': '#2E86AB', 'Decision Tree': '#D64545'}
for name, res in results.items():
    ax.plot(res['fpr'], res['tpr'], color=colors[name], linewidth=2.2,
            label=f"{name} (AUC = {res['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Random guess (AUC = 0.50)')
ax.set_title("ROC Curve Comparison")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc='lower right')
sns.despine()
plt.tight_layout()
plt.savefig('viz2_roc_curves.png', dpi=150)
plt.close()

# Viz 3: Feature importance (Decision Tree) / Coefficient magnitude (LogReg)
feat_names = (numeric_features +
              list(tree.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(categorical_features)))
importances = tree.named_steps['clf'].feature_importances_
imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp_df['feature'], imp_df['importance'], color='#4C956C')
ax.set_title("Top 10 Feature Importances (Decision Tree)")
ax.set_xlabel("Importance")
sns.despine()
plt.tight_layout()
plt.savefig('viz3_feature_importance.png', dpi=150)
plt.close()

# Viz 4: Learning curve for Logistic Regression - diagnose over/underfitting
train_sizes, train_scores, val_scores = learning_curve(
    log_reg, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE
)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='#2E86AB', label='Training accuracy')
ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                 train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color='#2E86AB')
ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='#D64545', label='Cross-validation accuracy')
ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color='#D64545')
ax.set_title("Learning Curve — Logistic Regression")
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Accuracy")
ax.legend(loc='lower right')
sns.despine()
plt.tight_layout()
plt.savefig('viz4_learning_curve.png', dpi=150)
plt.close()

print("\nAll visualizations saved.")

import json
summary = {name: {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                   for k, v in res.items() if k in ['acc', 'prec', 'rec', 'f1', 'roc_auc', 'cv_mean', 'cv_std', 'train_acc']}
           for name, res in results.items()}
with open('results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
