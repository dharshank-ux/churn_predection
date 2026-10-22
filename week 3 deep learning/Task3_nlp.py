import re, pandas as pd
import numpy as np

df = pd.read_csv("tickets.csv")
print(df.shape)
print(df.head())
print(df["category"].value_counts())
def clean(text):
    text = text.lower()                        # lowercase
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove punctuation
    return text.split()                        # tokenise into words

df["tokens"] = df["text"].apply(clean)
df["clean_text"] = df["tokens"].apply(" ".join)
print(df[["text", "clean_text"]].head(3))
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

X_tr, X_te, y_tr, y_te = train_test_split(
    df["clean_text"], df["category"], test_size=0.2,
    random_state=42, stratify=df["category"])

pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))])

t = time.time()
pipe.fit(X_tr, y_tr)
print("Training time (s):", round(time.time() - t, 3))

pred = pipe.predict(X_te)
print("Accuracy:", round(accuracy_score(y_te, pred), 3))
print("Macro F1:", round(f1_score(y_te, pred, average="macro"), 3))
print("Confusion matrix:")
print(confusion_matrix(y_te, pred, labels=pipe.classes_))
print("Class order:", list(pipe.classes_))
vec = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]
words = np.array(vec.get_feature_names_out())

for i, c in enumerate(clf.classes_):
    top = words[np.argsort(clf.coef_[i])[-10:][::-1]]
    print(c, "->", list(top))
# GloVe part (nearest words, king - man + woman, sentence vectors)
# was run in Google Colab because gensim does not install on Python 3.14.

    
