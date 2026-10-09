import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

X, y = make_classification(n_samples=5000, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

probas = {}
plt.figure(figsize=(7, 6))
print("Model | AUC | F1")
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    probas[name] = proba
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, (proba >= 0.5).astype(int))
    print(f"{name} | {auc:.3f} | {f1:.3f}")
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curves")
plt.legend()
plt.savefig("roc_curves.png")
plt.show()
from sklearn.metrics import precision_recall_curve, average_precision_score

# Part B: threshold for at least 80% recall (using XGBoost, the best model)
best_proba = probas["XGBoost"]
prec, rec, thr = precision_recall_curve(y_test, best_proba)

# prec and rec have one more value than thr, so drop the last one
ok = rec[:-1] >= 0.80
chosen = thr[ok][-1]          # highest threshold that still gives 80% recall
idx = ok.nonzero()[0][-1]
print(f"\nThreshold: {chosen:.3f}")
print(f"Recall at threshold   : {rec[idx]:.3f}")
print(f"Precision at threshold: {prec[idx]:.3f}")

# Part C: precision-recall curve for all 3 models
plt.figure(figsize=(7, 6))
for name, p in probas.items():
    pr, rc, _ = precision_recall_curve(y_test, p)
    ap = average_precision_score(y_test, p)
    plt.plot(rc, pr, label=f"{name} (AP={ap:.3f})")
plt.axhline(y_test.mean(), color="k", linestyle="--", label="Random guess")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.savefig("pr_curves.png")
plt.show()

# EXPLANATION: F1 vs AUC
# Use F1 when you have picked one threshold and care about the rare positive class.
# Use AUC to compare models across all thresholds, before choosing a threshold.
# ROC/AUC can look too optimistic on imbalanced data because true negatives dominate.
# The precision-recall curve is better for imbalanced data: it focuses only on the rare class.