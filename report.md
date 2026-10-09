# Customer Churn Prediction Report

## Problem
A telecom company loses customers every month. We predict which
customers are likely to churn so the retention team can contact
them early. About 27% of customers churn, so the data is imbalanced.

## Data
Telco Customer Churn dataset: about 7,000 customers with contract,
tenure, services and billing details. Split into train and a
held-out test set (1,409 customers). Added features: tenure_group,
avg_monthly_spend, num_services, has_internet.

## Method
- Models: Logistic Regression, Random Forest, XGBoost
  (XGBoost uses scale_pos_weight for imbalance)
- Tuned with RandomizedSearchCV (scoring = ROC-AUC, 5-fold CV)
- Threshold chosen to catch at least 75% of churners, because
  missing a churner costs more than a wasted retention offer
- Test set used once, at the very end

## Results
| Model | ROC-AUC | F1 |
|---|---|---|
| Logistic Regression | 0.846 | 0.592 |
| Random Forest | 0.847 | 0.563 |
| XGBoost | 0.846 | 0.629 |

Final test (chosen model: Random Forest, threshold 0.318):
- ROC-AUC: 0.844   F1: 0.635
- Confusion matrix: 798 stayed correctly, 237 false alarms,
  90 churners missed, 284 churners caught
- Recall 0.76, precision 0.55

Top 5 drivers: (write one line each from feature_importance.png)

## Limitations
- Only about 55% of flagged customers really churn
- Data is a single snapshot; behaviour may change over time
- Model predicts who may leave, not why or what offer will work

## Recommendations
1. Contact the flagged customers with a retention offer.
2. Push annual contracts for month-to-month customers.
3. Improve onboarding for new customers.
4. Retrain the model every few months and monitor it.