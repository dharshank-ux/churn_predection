import time, numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (AutoTokenizer,
    AutoModelForSequenceClassification, TrainingArguments,
    Trainer, DataCollatorWithPadding)

train = pd.read_csv("train.csv")
val = pd.read_csv("val.csv")
test = pd.read_csv("test.csv")

labels = sorted(train["category"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
for d in (train, val, test):
    d["label"] = d["category"].map(label2id)

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def prep(d):
    ds = Dataset.from_pandas(d[["text", "label"]].reset_index(drop=True))
    return ds.map(lambda b: tok(b["text"], truncation=True,
                  max_length=128), batched=True)

tr, va, te = prep(train), prep(val), prep(test)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(labels),
    id2label=id2label, label2id=label2id)

def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "macro_f1": f1_score(p.label_ids, preds, average="macro")}

args = TrainingArguments(output_dir="out", num_train_epochs=3,
    learning_rate=2e-5, per_device_train_batch_size=16,
    eval_strategy="epoch", save_strategy="no", report_to="none")

trainer = Trainer(model=model, args=args, train_dataset=tr,
    eval_dataset=va, compute_metrics=metrics,
    data_collator=DataCollatorWithPadding(tok))

t = time.time()
t0 = time.time()
trainer.train()
print("TRAIN TIME (s):", round(time.time() - t0, 1))
print("TRAINING TIME (s):", round(time.time() - t, 1))

print("VAL:", trainer.evaluate())
print("TEST:", trainer.predict(te).metrics)

# inference time per ticket (100 tickets, one at a time)
model.eval()
texts = val["text"].tolist()[:100]
t = time.time()
for x in texts:
    inp = tok(x, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        model(**inp)
print("INFERENCE (ms/ticket):", round((time.time() - t) / 100 * 1000, 1))

trainer.save_model("model_category/")
tok.save_pretrained("model_category/")