## Data
- Source: synthetic tickets generated with make_tickets.py
- Rows: 700 (after cleaning)
- Category counts: billing 135, technical 127, shipping 135, account 156, refund 147
- Priority counts: high 288, low 229, medium 183
- Cleaning: lowercase, trim, remove duplicates
- Split: 70/15/15 train/val/test (490/105/105), stratified on category + priority

## Baseline results (validation set)
- Category: accuracy 1.0, macro F1 1.0
- Priority: accuracy 1.0, macro F1 1.0
- Confusion matrices: cm_category.png, cm_priority.png
- Saved models: baseline_category.joblib, baseline_priority.joblib
- Note: scores are perfect because the data is synthetic (made from templates).
- GloVe averaged sentence vectors + Logistic Regression: accuracy 1.0, macro F1 1.0 (run in Colab)
- TF-IDF: accuracy 1.0, macro F1 1.0. Both perfect because the tickets are template-generated; TF-IDF is simpler and faster, so I would choose it.

## Task 4: Comparison

| Model | Accuracy | Macro F1 | Training time | Inference time |
|---|---|---|---|---|
| TF-IDF + LogReg | 1.0 | 1.0 | 0.19 s (CPU) | 0.046 ms/text |
| DistilBERT | 1.0 | 1.0 | 9.9 s (T4 GPU) | 1.879 ms/text |

Both score 1.0 because the tickets are template-generated. TF-IDF is much faster to train and run, so I would choose TF-IDF for this data.

## BERT vs GPT
- BERT is an encoder-only model that reads the whole sentence in both directions.
- GPT is a decoder-only model that reads left to right and predicts the next word.
- BERT is better for understanding tasks like classification, sentiment and NER.
- GPT is better for generating text, such as chat, writing and summaries.
- For this task (ticket classification) I would use BERT/DistilBERT, or TF-IDF since the data is simple.

## API
POST /predict returns label and confidence. Empty text returns 422 ("Text must not be empty"). Labels: LABEL_0 account, LABEL_1 billing, LABEL_2 refund, LABEL_3 shipping, LABEL_4 technical.