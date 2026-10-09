import sys
import joblib
import pandas as pd


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


path = sys.argv[1]
df = pd.read_csv(path)

ids = df["customerID"] if "customerID" in df.columns else df.index

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = add_features(df)

model = joblib.load("model.joblib")
threshold = joblib.load("threshold.joblib")

df = df[list(model.feature_names_in_)]

proba = model.predict_proba(df)[:, 1]
out = pd.DataFrame({
    "customerID": ids,
    "churn_probability": proba.round(4),
    "churn_prediction": (proba >= threshold).astype(int),
})
out.to_csv("predictions.csv", index=False)
print("Saved predictions.csv with", len(out), "rows")