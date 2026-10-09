import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load and split 80/20
X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Models
models = {
    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

# 3. Train and evaluate
rows, preds = [], {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    preds[name] = p
    rows.append([name,
                 mean_absolute_error(y_te, p),
                 np.sqrt(mean_squared_error(y_te, p)),
                 r2_score(y_te, p)])

table = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"]).round(3)
print(table.to_string(index=False))

# 4. Plots
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for i, name in enumerate(["LinearRegression", "RandomForest"]):
    p = preds[name]
    axes[0, i].scatter(y_te, p, s=5, alpha=0.4)
    axes[0, i].plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "r--")
    axes[0, i].set_title(f"{name}: Predicted vs Actual")
    axes[0, i].set_xlabel("Actual")
    axes[0, i].set_ylabel("Predicted")

    axes[1, i].scatter(p, y_te - p, s=5, alpha=0.4)
    axes[1, i].axhline(0, color="r", linestyle="--")
    axes[1, i].set_title(f"{name}: Residuals")
    axes[1, i].set_xlabel("Predicted")
    axes[1, i].set_ylabel("Residual")

plt.tight_layout()
plt.savefig("regression_plots.png")
plt.show()

# EXPLANATION
# MAE is the average absolute error, in the same units as the target.
# RMSE squares errors first, so it punishes large errors more than MAE.
# RMSE is always >= MAE; a big gap means some very large errors.
# R-squared of 0.6 means the model explains 60% of the variance in the target.