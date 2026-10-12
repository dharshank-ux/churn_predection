import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

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

# Model 1 and 2: Logistic Regression and Decision Tree
log_pipe = Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=2000))])
tree_pipe = Pipeline([("prep", make_prep()), ("model", DecisionTreeClassifier(max_depth=5, random_state=42))])

log_pipe.fit(X_train, y_train)
tree_pipe.fit(X_train, y_train)

print("Logistic Regression val:", round(log_pipe.score(X_val, y_val), 3))
print("Decision Tree val:      ", round(tree_pipe.score(X_val, y_val), 3))
from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ("prep", make_prep()),
    ("model", RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)),
])
rf_pipe.fit(X_train, y_train)

print("Random Forest OOB score:", round(rf_pipe.named_steps["model"].oob_score_, 3))
print("Random Forest val:      ", round(rf_pipe.score(X_val, y_val), 3))
from xgboost import XGBClassifier

# Preprocess by hand so we can give XGBoost a validation set
prep_xgb = make_prep()
Xtr_t = prep_xgb.fit_transform(X_train)
Xva_t = prep_xgb.transform(X_val)

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    early_stopping_rounds=20,
    eval_metric="logloss",
    random_state=42,
)
xgb.fit(Xtr_t, y_train, eval_set=[(Xva_t, y_val)], verbose=False)

print("XGBoost best round:", xgb.best_iteration)
print("XGBoost val:       ", round(xgb.score(Xva_t, y_val), 3))
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
    "Validation score": [
        log_pipe.score(X_val, y_val),
        tree_pipe.score(X_val, y_val),
        rf_pipe.score(X_val, y_val),
        xgb.score(Xva_t, y_val),
    ],
})
print(results.round(3))
results.to_csv("assessment_results.csv", index=False)
# Probabilities for 5 validation rows
proba = log_pipe.predict_proba(X_val.head(5))
pred = log_pipe.predict(X_val.head(5))

out = pd.DataFrame({
    "P(stay)": proba[:, 0].round(3),
    "P(churn)": proba[:, 1].round(3),
    "predict": pred,
})
print(out)
# Feature names after preprocessing
names = rf_pipe.named_steps["prep"].get_feature_names_out()

rf_imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=names)
print("Top 5 Random Forest features:")
print(rf_imp.sort_values(ascending=False).head(5).round(3))

xgb_names = prep_xgb.get_feature_names_out()
xgb_imp = pd.Series(xgb.feature_importances_, index=xgb_names)
print("Top 5 XGBoost features:")
print(xgb_imp.sort_values(ascending=False).head(5).round(3))