import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Shape:", df.shape)
print("\nMissing values:")
print(df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nChurn balance:")
print(df["Churn"].value_counts(normalize=True))
from sklearn.model_selection import train_test_split

df["Churn"] = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)
print("Churn rate in train:", y_train.mean())
print("Churn rate in test:", y_test.mean())
import matplotlib.pyplot as plt
import seaborn as sns

# EDA uses TRAINING data only
train = X_train.copy()
train["Churn"] = y_train

# Chart 1: churn rate by contract type
rate = train.groupby("Contract")["Churn"].mean().sort_values()
sns.barplot(x=rate.index, y=rate.values)
plt.title("Churn rate by Contract")
plt.ylabel("Churn rate")
plt.savefig("chart1_contract.png")

# Chart 2: tenure of customers who stayed vs left
plt.figure()
sns.histplot(data=train, x="tenure", hue="Churn", bins=30, multiple="stack")
plt.title("Tenure by Churn (0 = stayed, 1 = left)")
plt.savefig("chart2_tenure.png")

# Chart 3: monthly charges for customers who stayed vs left
plt.figure()
sns.boxplot(data=train, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn (0 = stayed, 1 = left)")
plt.savefig("chart3_monthly.png")

# Chart 4: churn rate by internet service
plt.figure()
rate = train.groupby("InternetService")["Churn"].mean().sort_values()
sns.barplot(x=rate.index, y=rate.values)
plt.title("Churn rate by Internet Service")
plt.ylabel("Churn rate")
plt.savefig("chart4_internet.png")

# Chart 5: churn rate by payment method
plt.figure(figsize=(9, 5))
rate = train.groupby("PaymentMethod")["Churn"].mean().sort_values()
sns.barplot(x=rate.index, y=rate.values)
plt.title("Churn rate by Payment Method")
plt.ylabel("Churn rate")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("chart5_payment.png")

# ASSUMPTIONS AND METRIC
# Assumptions:
# 1. Blank TotalCharges values are new customers (tenure 0), so they are treated as missing and imputed.
# 2. customerID is only an identifier and carries no signal, so it is dropped.
# 3. The 20% test set is held out now and used only at the final step.
# 4. All EDA and model choices use training data only.
#
# Metric I will optimise: ROC-AUC (also tracking F1).
# Why: only about 27% of customers churn, so accuracy is misleading
# (always predicting "stay" gives 73%). ROC-AUC measures how well the model
# ranks likely churners above non-churners across all thresholds.
# F1 balances precision and recall for the churn class.

# TASK 2: FEATURE ENGINEERING
def add_features(d):
    d = d.copy()
    d["tenure_group"] = pd.cut(d["tenure"], [-1, 12, 24, 48, 72],
                               labels=["0-1y", "1-2y", "2-4y", "4y+"]).astype(str)
    d["avg_monthly_spend"] = d["TotalCharges"] / d["tenure"].replace(0, 1)
    services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
    d["num_services"] = (d[services] == "Yes").sum(axis=1)
    d["has_internet"] = (d["InternetService"] != "No").astype(int)
    return d

X_train_fe = add_features(X_train)
print(X_train_fe[["tenure", "tenure_group", "avg_monthly_spend",
                  "num_services", "has_internet"]].head())

# PREPROCESSING + MODELS
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

num_cols = X_train_fe.select_dtypes("number").columns
cat_cols = X_train_fe.select_dtypes(exclude="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

models = {
    "Dummy (bonus)": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
}

rows = []
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)])
    r = cross_validate(pipe, X_train_fe, y_train, cv=5,
                       scoring=["roc_auc", "f1"])
    rows.append([name, round(r["test_roc_auc"].mean(), 3),
                 round(r["test_f1"].mean(), 3)])

results = pd.DataFrame(rows, columns=["Model", "CV ROC-AUC", "CV F1"])
print("\n", results)
results.to_csv("results.csv", index=False)

# CONCLUSION
# Logistic Regression is the best baseline (CV ROC-AUC 0.846, F1 0.589).
# Random Forest is slightly lower (0.825, 0.551) with default settings.
# Both clearly beat the Dummy baseline (AUC 0.50), so the features carry signal.
# The test set has NOT been used yet; it is reserved for the final step.
# Next: tune the models and evaluate the best one once on the test set.