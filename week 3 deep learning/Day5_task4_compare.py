import time, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

pipe = Pipeline([("tfidf", TfidfVectorizer()),
                 ("clf", LogisticRegression(max_iter=1000))])

t = time.time()
pipe.fit(train["text"], train["category"])
train_time = time.time() - t

t = time.time()
pred = pipe.predict(test["text"])
infer_ms = (time.time() - t) / len(test) * 1000

print("accuracy:", accuracy_score(test["category"], pred))
print("macro F1:", f1_score(test["category"], pred, average="macro"))
print("training time (s):", round(train_time, 2))
print("inference (ms per text):", round(infer_ms, 3))