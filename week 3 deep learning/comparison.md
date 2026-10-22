# Model Comparison: Category Classifier

| Model | Accuracy | Macro F1 | Training time | Inference/ticket |
|---|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 1.0 | 1.0 | 2.1 s | 1.19 ms |
| DistilBERT | 1.0 | 1.0 | 120.7 s | 51.1 ms |

## Justification
1. Both models score 1.0 accuracy and 1.0 macro F1, so DistilBERT gives no accuracy gain on this data.
2. DistilBERT wins on messy, unseen wording, but these tickets are short and templated, so keywords are enough.
3. The baseline trains about 57 times faster (2.1 s vs 120.7 s).
4. The baseline predicts about 43 times faster (1.19 ms vs 51.1 ms) and runs on a plain CPU with no big model, so it costs less.
5. Decision: deploy the baseline. Switch to DistilBERT only if real tickets become more varied and baseline accuracy drops.