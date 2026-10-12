# Week 2 ML Assessment: Telco Customer Churn

## Dataset
Telco Customer Churn (7043 rows). Target: Churn (1 = left).
Split 60/20/20, stratified, random_state=42.

## Task 1: Splits and leakage
- Decision tree overfits: depth None train 1.00, val 0.72.
- 5-fold CV (Logistic Regression): 0.81 +/- 0.01.
- Scaling before the split leaks test statistics. Scores were equal (0.807), but the correct way is to fit on train only.

## Task 2: Pipeline
- New features: tenure_years, tenure_months_left, charge_ratio, log_total_charges.
- ColumnTransformer inside a Pipeline (imputing, scaling, one-hot, ordinal for Contract).
- Pipeline CV accuracy: 0.80. Saved with joblib (churn_pipeline.joblib).

## Task 3: Models
| Model | Validation accuracy |
|---|---|
| Logistic Regression | 0.806 |
| Decision Tree | 0.782 |
| Random Forest | 0.784 |
| XGBoost | 0.797 |

- Regression target: MonthlyCharges. Lasso set 25 of 45 coefficients to zero.
- K-Means with k = 3 on tenure, MonthlyCharges and TotalCharges.

## Task 4: Evaluation and tuning
- Accuracy misleads because about 73% of customers stay.
- Threshold 0.29 gives recall 0.75 at precision 0.53.
- Random Forest tuned: best CV AUC 0.847 (max_depth 5, min_samples_leaf 5, n_estimators 200).
- XGBoost tuned: best CV AUC 0.849 (subsample 0.7, n_estimators 200, max_depth 3, learning_rate 0.01).
- Validation AUC: Random Forest 0.837, XGBoost 0.839.
- Best model: XGBoost (tuned) | Test ROC-AUC 0.84 | F1 0.51
- Test set: accuracy 0.795, precision 0.694, recall 0.406.
- Regression (Ridge vs mean baseline): Ridge MAE 0.78, RMSE 1.03, R2 0.999. Mean baseline MAE 26.84, RMSE 30.58, R2 0.00.

## Decisions
- Telco has no loan amount or date, so I used MonthlyCharges and tenure instead.
- The test set was used once, at the very end.
- charge_ratio was removed for the regression task because it is built from the target.
- Ridge R2 is very high because TotalCharges is almost MonthlyCharges times tenure, so the target is easy to rebuild from other columns.
- At the default 0.5 threshold recall is only 0.41. Lowering the threshold to about 0.29 would catch more churners, at the cost of more false alarms.