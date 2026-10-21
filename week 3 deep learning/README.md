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