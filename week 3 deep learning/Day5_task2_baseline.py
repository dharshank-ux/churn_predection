import time
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")


def run(label):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "clf__C": [0.1, 1, 10, 100],
    }
    gs = GridSearchCV(pipe, grid, cv=5, scoring="f1_macro")

    t0 = time.time()
    gs.fit(train["text"], train[label])
    print("BASELINE TRAIN TIME (s):", round(time.time() - t0, 1))

    model = gs.best_estimator_
    print("\n==", label, "==")
    print("best settings:", gs.best_params_)

    t1 = time.time()
    for x in val["text"][:100]:
        model.predict([x])
    print("BASELINE INFERENCE (ms/ticket):",
          round((time.time() - t1) / 100 * 1000, 2))

    pred = model.predict(val["text"])
    print("accuracy:", accuracy_score(val[label], pred))
    print("macro F1:", f1_score(val[label], pred, average="macro"))

    for text, true, p in list(zip(val["text"], val[label], pred))[:10]:
        print(f"{text} | true: {true} | pred: {p}")

    disp = ConfusionMatrixDisplay.from_predictions(val[label], pred)
    disp.figure_.savefig(f"cm_{label}.png")
    joblib.dump(model, f"baseline_{label}.joblib")


run("category")
run("priority")