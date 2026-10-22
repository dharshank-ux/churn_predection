import collections, numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, pipeline)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Task 1: data
ds = load_dataset("dair-ai/emotion")
data = ds["train"].shuffle(seed=42).select(range(2000))
split = data.train_test_split(test_size=0.2, seed=42)
train_ds, val_ds = split["train"], split["test"]
names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
print("Train size:", len(train_ds), "Val size:", len(val_ds))
print("Train labels:", collections.Counter(train_ds["label"]))
print("Val labels:", collections.Counter(val_ds["label"]))

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(b):
    return tok(b["text"], truncation=True, padding="max_length", max_length=64)
train_tok = train_ds.map(tokenize, batched=True)
val_tok = val_ds.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=6)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="weighted")}

args = TrainingArguments(output_dir="out", learning_rate=5e-5,
    num_train_epochs=3, per_device_train_batch_size=16,
    per_device_eval_batch_size=16, eval_strategy="epoch",
    save_strategy="no", logging_strategy="epoch", report_to="none")

# Task 2: train and evaluate
trainer = Trainer(model=model, args=args, train_dataset=train_tok,
                  eval_dataset=val_tok, compute_metrics=compute_metrics)
trainer.train()
for log in trainer.state.log_history:
    if "eval_accuracy" in log:
        print("Epoch", log["epoch"], "accuracy", round(log["eval_accuracy"], 3),
              "F1", round(log["eval_f1"], 3))

out = trainer.predict(val_tok)
preds = np.argmax(out.predictions, axis=1)
print("Confusion matrix:\n", confusion_matrix(out.label_ids, preds))
ft_acc = accuracy_score(out.label_ids, preds)
ft_f1 = f1_score(out.label_ids, preds, average="weighted")

# Save and reload
trainer.save_model("my_model"); tok.save_pretrained("my_model")
clf = pipeline("text-classification", model="my_model", tokenizer="my_model")
tests = ["I am so happy today", "This is terrifying",
         "I miss my old friends so much", "How dare you lie to me",
         "Wow, I did not expect that at all"]
for s in tests:
    r = clf(s)[0]
    idx = int(r["label"].split("_")[1])
    print(f"{s} -> {names[idx]} ({r['score']:.2f})")

# Baseline
vec = TfidfVectorizer()
lr = LogisticRegression(max_iter=1000).fit(vec.fit_transform(train_ds["text"]), train_ds["label"])
base = accuracy_score(val_ds["label"], lr.predict(vec.transform(val_ds["text"])))
print(f"Fine-tuned: accuracy {ft_acc:.2f}, F1 {ft_f1:.2f} | TF-IDF baseline: accuracy {base:.2f}")