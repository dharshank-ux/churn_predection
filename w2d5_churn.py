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
plt.show()
# Chart 2: tenure of customers who stayed vs left
plt.figure()
sns.histplot(data=train, x="tenure", hue="Churn", bins=30, multiple="stack")
plt.title("Tenure by Churn (0 = stayed, 1 = left)")
plt.savefig("chart2_tenure.png")
plt.show()
# Chart 3: monthly charges for customers who stayed vs left
plt.figure()
sns.boxplot(data=train, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn (0 = stayed, 1 = left)")
plt.savefig("chart3_monthly.png")
plt.show()
# Chart 4: churn rate by internet service
plt.figure()
rate = train.groupby("InternetService")["Churn"].mean().sort_values()
sns.barplot(x=rate.index, y=rate.values)
plt.title("Churn rate by Internet Service")
plt.ylabel("Churn rate")
plt.savefig("chart4_internet.png")
plt.show()
# Chart 5: churn rate by payment method
plt.figure(figsize=(9, 5))
rate = train.groupby("PaymentMethod")["Churn"].mean().sort_values()
sns.barplot(x=rate.index, y=rate.values)
plt.title("Churn rate by Payment Method")
plt.ylabel("Churn rate")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("chart5_payment.png")
plt.show()