import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Fix the TotalCharges column (it has blank cells)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# Target: 1 = churned, 0 = stayed
y = (df["Churn"] == "Yes").astype(int)

# Features, with words turned into numbers
X = df.drop(columns=["Churn", "customerID"])
X = pd.get_dummies(X, drop_first=True)

# Split 60/20/20
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# Overfitting demo
depths = [1, 3, 5, 10, None]
train_scores = []
val_scores = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    tr = model.score(X_train, y_train)
    va = model.score(X_val, y_val)
    train_scores.append(tr)
    val_scores.append(va)
    print(f"Depth {d}: train {tr:.2f} val {va:.2f}")

labels = [str(d) for d in depths]
plt.plot(labels, train_scores, marker="o", label="train")
plt.plot(labels, val_scores, marker="o", label="validation")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("overfitting.png")
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

scores = cross_val_score(LogisticRegression(max_iter=2000), X_train, y_train, cv=5)
print("CV scores:", scores)
print(f"CV: {scores.mean():.2f} +/- {scores.std():.2f}")
from sklearn.preprocessing import StandardScaler

# WRONG way: scale ALL the data first, then split
X_scaled_all = StandardScaler().fit_transform(X)
a_tr, a_te, b_tr, b_te = train_test_split(
    X_scaled_all, y, test_size=0.2, stratify=y, random_state=42
)
leaky_model = LogisticRegression(max_iter=5000).fit(a_tr, b_tr)
leaky_score = leaky_model.score(a_te, b_te)

# RIGHT way: split first, learn scaling from train only
c_tr, c_te, d_tr, d_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler().fit(c_tr)
correct_model = LogisticRegression(max_iter=5000).fit(scaler.transform(c_tr), d_tr)
correct_score = correct_model.score(scaler.transform(c_te), d_te)

print("Leaky score:  ", round(leaky_score, 4))
print("Correct score:", round(correct_score, 4))
# Explanation of leakage:
# 1. Scaling before the split calculates mean and std using the test rows too.
# 2. So information from the test set leaks into the training process.
# 3. The correct way fits the scaler on train only, then only transforms the test data.
# 4. Leakage makes scores look better than they will be on truly new data.
# ---------- Supervised vs Unsupervised ----------
# 1. Spam detection: SUPERVISED (emails are labeled spam / not spam)
# 2. Customer segmentation: UNSUPERVISED (no labels, we find groups, e.g. K-Means)
# 3. House price prediction: SUPERVISED (regression, the price is the label)
# 4. Anomaly detection: UNSUPERVISED (usually no labels for rare cases)

# ---------- ML workflow in 7 steps ----------
# 1. Define the problem (e.g. predict which customers will churn)
# 2. Collect the data
# 3. Clean and explore the data
# 4. Split into train, validation and test
# 5. Preprocess and engineer features
# 6. Train and tune models
# 7. Evaluate on the test set, then deploy and monitor

