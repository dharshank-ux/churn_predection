import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn", "customerID"])

def add_features(data):
    data = data.copy()
    data["tenure_years"] = data["tenure"] // 12
    data["tenure_months_left"] = data["tenure"] % 12
    data["charge_ratio"] = data["MonthlyCharges"] / (data["TotalCharges"] + 1)
    data["log_total_charges"] = np.log1p(data["TotalCharges"])
    return data

X = add_features(X)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

ordinal_cols = ["Contract"]
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols and c not in ordinal_cols]

def make_prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oe", OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]]))]), ordinal_cols),
    ])

models = {
    "Logistic Regression": Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=2000))]),
    "Decision Tree": Pipeline([("prep", make_prep()), ("model", DecisionTreeClassifier(max_depth=5, random_state=42))]),
    "Random Forest": Pipeline([("prep", make_prep()), ("model", RandomForestClassifier(n_estimators=200, random_state=42))]),
}

for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_val)
    print("\n" + name)
    print("Accuracy :", round(accuracy_score(y_val, pred), 3))
    print("Precision:", round(precision_score(y_val, pred), 3))
    print("Recall   :", round(recall_score(y_val, pred), 3))
    print("F1       :", round(f1_score(y_val, pred), 3))
    print(confusion_matrix(y_val, pred))

# Why accuracy misleads on imbalanced classes:
# About 73% of customers do not churn. A model that always says "no churn"
# scores about 73% accuracy but catches 0% of the churners (recall = 0).
# So we also look at precision, recall and F1.
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.figure(figsize=(6, 5))
for name, m in models.items():
    proba = m.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, proba)
    auc = roc_auc_score(y_val, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")
    print(name, "ROC-AUC:", round(auc, 3))

plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)")
plt.title("ROC curves")
plt.legend()
plt.savefig("assessment_roc.png")
plt.show()
from sklearn.metrics import precision_recall_curve

best = models["Logistic Regression"]
proba = best.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, proba)

# Highest threshold where recall is still at least 0.75
ok = np.where(rec[:-1] >= 0.75)[0]
i = ok[-1]
print(f"Threshold {thr[i]:.2f}: recall {rec[i]:.2f}, precision {prec[i]:.2f}")

pred_t = (proba >= thr[i]).astype(int)
print(confusion_matrix(y_val, pred_t))
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

# Random forest: 3 parameters
rf_grid = GridSearchCV(
    Pipeline([("prep", make_prep()), ("model", RandomForestClassifier(random_state=42))]),
    param_grid={
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None],
        "model__min_samples_leaf": [1, 5],
    },
    cv=3, scoring="roc_auc", n_jobs=-1,
)
rf_grid.fit(X_train, y_train)
print("RF best params:", rf_grid.best_params_)
print("RF best CV AUC:", round(rf_grid.best_score_, 3))

# XGBoost: 15 random combinations
xgb_search = RandomizedSearchCV(
    Pipeline([("prep", make_prep()), ("model", XGBClassifier(eval_metric="logloss", random_state=42))]),
    param_distributions={
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 1.0],
    },
    n_iter=15, cv=3, scoring="roc_auc", n_jobs=-1, random_state=42,
)
xgb_search.fit(X_train, y_train)
print("XGB best params:", xgb_search.best_params_)
print("XGB best CV AUC:", round(xgb_search.best_score_, 3))
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

drop = ["MonthlyCharges", "charge_ratio"]
yr_train, yr_val = X_train["MonthlyCharges"], X_val["MonthlyCharges"]
Xr_train, Xr_val = X_train.drop(columns=drop), X_val.drop(columns=drop)
num_r = [c for c in num_cols if c not in drop]

prep_r = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_r),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oe", OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]]))]), ordinal_cols),
])

reg = Pipeline([("prep", prep_r), ("model", Ridge(alpha=1.0))]).fit(Xr_train, yr_train)
base = DummyRegressor(strategy="mean").fit(Xr_train, yr_train)

for label, m in [("Ridge", reg), ("Mean baseline", base)]:
    p = m.predict(Xr_val)
    print(f"{label}: MAE {mean_absolute_error(yr_val, p):.2f}  "
          f"RMSE {np.sqrt(mean_squared_error(yr_val, p)):.2f}  "
          f"R2 {r2_score(yr_val, p):.3f}")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

candidates = {"Random Forest (tuned)": rf_grid.best_estimator_, "XGBoost (tuned)": xgb_search.best_estimator_}
val_auc = {n: roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]) for n, m in candidates.items()}
print("Validation AUC:", {k: round(v, 3) for k, v in val_auc.items()})

best_name = max(val_auc, key=val_auc.get)
final = candidates[best_name]

# The one and only look at the test set
proba_test = final.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)
print(f"Best model: {best_name} | ROC-AUC {roc_auc_score(y_test, proba_test):.2f} | F1 {f1_score(y_test, pred_test):.2f}")
print("Test accuracy :", round(accuracy_score(y_test, pred_test), 3))
print("Test precision:", round(precision_score(y_test, pred_test), 3))
print("Test recall   :", round(recall_score(y_test, pred_test), 3))
print(confusion_matrix(y_test, pred_test))