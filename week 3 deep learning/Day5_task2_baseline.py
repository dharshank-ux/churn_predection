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
    gs.fit(train["text"], train[label])
    model = gs.best_estimator_
    print("\n==", label, "==")
    print("best settings:", gs.best_params_)

    pred = model.predict(val["text"])
    print("accuracy:", accuracy_score(val[label], pred))
    print("macro F1:", f1_score(val[label], pred, average="macro"))

    disp = ConfusionMatrixDisplay.from_predictions(val[label], pred, xticks_rotation=45)
    disp.figure_.savefig("cm_" + label + ".png", bbox_inches="tight")

    # 10 most confusing = lowest confidence in the true label
    proba = model.predict_proba(val["text"])
    classes = list(model.classes_)
    true_idx = [classes.index(t) for t in val[label]]
    true_conf = proba[np.arange(len(val)), true_idx]
    worst = np.argsort(true_conf)[:10]
    for i in worst:
        print(val["text"].iloc[i][:80], "| true:", val[label].iloc[i], "| pred:", pred[i])

    joblib.dump(model, "baseline_" + label + ".joblib")

run("category")
run("priority")