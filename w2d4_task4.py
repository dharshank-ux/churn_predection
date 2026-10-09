import time
import pandas as pd
from scipy.stats import randint
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Grid search
grid = {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_leaf": [1, 2, 4]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5, scoring="f1", n_jobs=-1)
t = time.time()
gs.fit(X_tr, y_tr)
grid_time = time.time() - t

# Randomized search
dist = {"n_estimators": randint(50, 500), "max_depth": randint(3, 30), "min_samples_leaf": randint(1, 10)}
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), dist, n_iter=20, cv=5,
                        scoring="f1", n_jobs=-1, random_state=42)
t = time.time()
rs.fit(X_tr, y_tr)
rand_time = time.time() - t

# Results
print("Grid best_params_:", gs.best_params_)
print("Grid best_score_:", round(gs.best_score_, 3))
print("Random best_params_:", rs.best_params_)
print("Random best_score_:", round(rs.best_score_, 3))

cols = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
print(pd.DataFrame(gs.cv_results_)[cols].sort_values("rank_test_score").head(10).to_string())

print(f"Grid search: best F1 {gs.best_score_:.2f} in {grid_time:.0f} s | "
      f"Randomized: best F1 {rs.best_score_:.2f} in {rand_time:.0f} s")

# Final test evaluation, done ONCE
best = gs.best_estimator_ if gs.best_score_ >= rs.best_score_ else rs.best_estimator_
print("Test F1:", round(f1_score(y_te, best.predict(X_te)), 3))

# WHY NOT TUNE ON THE TEST SET
# Tuning on the test set leaks information, so the score becomes optimistically biased.
# The test set must act like unseen data, and it stops doing that once you choose settings from it.
# Tune with cross-validation on the training data and use the test set once at the end.