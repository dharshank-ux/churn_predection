from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
NAMES = {"LABEL_0": "account", "LABEL_1": "billing", "LABEL_2": "refund",
         "LABEL_3": "shipping", "LABEL_4": "technical"}

# Loaded once, when the server starts
clf = pipeline("text-classification", model="new_model/saved_model")

class Ticket(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Ticket):
    if not item.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty")
    out = clf(item.text)[0]
    return {"label": NAMES.get(out["label"], out["label"]), "confidence": round(out["score"], 4)}