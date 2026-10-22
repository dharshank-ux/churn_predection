import time, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("train.csv")
pipe = Pipeline([("tfidf", TfidfVectorizer()),
                 ("clf", LogisticRegression(max_iter=1000))])
t = time.time()
pipe.fit(train["text"], train["category"])
print("BASELINE training time (s):", round(time.time() - t, 2))