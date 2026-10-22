from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

BASE = Path(__file__).resolve().parent.parent
models = {}

@asynccontextmanager
async def lifespan(app):
    models["category"] = joblib.load(BASE / "baseline_category.joblib")
    models["priority"] = joblib.load(BASE / "baseline_priority.joblib")
    yield

app = FastAPI(title="Ticket Classifier", lifespan=lifespan)

class Ticket(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("text must not be empty")
        return v

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(ticket: Ticket):
    cat_model = models["category"]
    probs = cat_model.predict_proba([ticket.text])[0]
    i = probs.argmax()
    return {
        "category": str(cat_model.classes_[i]),
        "priority": str(models["priority"].predict([ticket.text])[0]),
        "confidence": round(float(probs[i]), 2),
    }