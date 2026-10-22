import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tickets.csv")[["text", "category", "priority"]].dropna()
print("rows before cleaning:", len(df))

df["text"] = df["text"].str.lower().str.strip()
df = df.drop_duplicates(subset="text")

print("rows after cleaning:", len(df))
print(df["category"].value_counts())
print(df["priority"].value_counts())

strat = df["category"] + "_" + df["priority"]
train, temp = train_test_split(df, test_size=0.30, stratify=strat, random_state=42)
val, test = train_test_split(
    temp, test_size=0.50,
    stratify=temp["category"] + "_" + temp["priority"],
    random_state=42,
)

print("train/val/test:", len(train), len(val), len(test))
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)