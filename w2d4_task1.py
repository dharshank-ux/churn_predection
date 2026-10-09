import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Create imbalanced data (95% normal, 5% fraud)
X, y = make_classification(
    n_samples=5000, weights=[0.95, 0.05], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Fraud share:", y.mean())

# 2. Model that always predicts the majority class
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
pred = dummy.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall   :", recall_score(y_test, pred))
print("F1       :", f1_score(y_test, pred))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 3. Normal Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print(classification_report(y_test, pred_lr))
cm = confusion_matrix(y_test, pred_lr)
print("Confusion matrix:\n", cm)
tn, fp, fn, tp = cm.ravel()
print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)

# 5. With class_weight='balanced'
lr_bal = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_bal.fit(X_train, y_train)
pred_bal = lr_bal.predict(X_test)

print("\n--- Logistic Regression (balanced) ---")
print(classification_report(y_test, pred_bal))
print("Confusion matrix:\n", confusion_matrix(y_test, pred_bal))
print("Recall  normal  :", recall_score(y_test, pred_lr))
print("Recall  balanced:", recall_score(y_test, pred_bal))
print("F1      normal  :", f1_score(y_test, pred_lr))
print("F1      balanced:", f1_score(y_test, pred_bal))

# EXPLANATION (fraud example)
# TP: fraud transaction, model says fraud (fraud caught)
# FP: legit transaction, model says fraud (customer blocked by mistake)
# FN: fraud transaction, model says legit (fraud missed, the costly one)
# TN: legit transaction, model says legit (correct)
#
# class_weight='balanced' raised recall (0.36 -> 0.78),
# so more frauds are caught (43 of 55 instead of 20).
# But F1 dropped (0.47 -> 0.36) because false alarms rose
# (FP = 142), which lowers precision.