import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn", "customerID"])

# Same split as Task 1 (so results are comparable)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

print(X_train.shape)
print(X_train.isna().sum()[X_train.isna().sum() > 0])
def add_features(data):
    data = data.copy()
    # 1. Date-style decomposition of tenure
    data["tenure_years"] = data["tenure"] // 12
    data["tenure_months_left"] = data["tenure"] % 12
    # 2. Interaction (ratio)
    data["charge_ratio"] = data["MonthlyCharges"] / (data["TotalCharges"] + 1)
    # 3. Log transform for a skewed column
    data["log_total_charges"] = np.log1p(data["TotalCharges"])
    return data

X_train = add_features(X_train)
X_val = add_features(X_val)
X_test = add_features(X_test)

print(X_train.shape)
print(X_train[["tenure", "tenure_years", "tenure_months_left", "charge_ratio", "log_total_charges"]].head())
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Sort the columns into 3 groups
ordinal_cols = ["Contract"]
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols and c not in ordinal_cols]

# Numbers: fill blanks with median, then scale
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Words: fill blanks with most frequent, then one-hot encode
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Ordered words: Contract has a natural order
ord_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]])),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
    ("ord", ord_pipe, ordinal_cols),
])

# Full pipeline: preprocessing + model
pipe = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=2000)),
])

scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f"Pipeline CV accuracy: {scores.mean():.2f}")
from sklearn.impute import KNNImputer

# Same pipeline, but numbers are filled using the 5 most similar customers
num_pipe_knn = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler()),
])

preprocess_knn = ColumnTransformer([
    ("num", num_pipe_knn, num_cols),
    ("cat", cat_pipe, cat_cols),
    ("ord", ord_pipe, ordinal_cols),
])

pipe_knn = Pipeline([
    ("prep", preprocess_knn),
    ("model", LogisticRegression(max_iter=2000)),
])

scores_knn = cross_val_score(pipe_knn, X_train, y_train, cv=5)
print(f"Median imputer CV: {scores.mean():.3f}")
print(f"KNN imputer CV:    {scores_knn.mean():.3f}")
import joblib

# Train the final pipeline on the training data, then save it
pipe.fit(X_train, y_train)
joblib.dump(pipe, "churn_pipeline.joblib")

# Load it back, like a different program would
loaded = joblib.load("churn_pipeline.joblib")

# 3 new raw rows (take them from the validation set)
new_rows = X_val.head(3)
print("Predictions for 3 new rows:", loaded.predict(new_rows))
print("Actual answers:            ", y_val.head(3).values)

