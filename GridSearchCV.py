import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------------
# 1. Create example dataset
# ------------------------------------------------------------

rng = np.random.default_rng(42)

n_samples = 300

data = pd.DataFrame({
    "age": rng.integers(21, 65, size=n_samples),
    "income": rng.normal(6000, 2000, size=n_samples),
    "city": rng.choice(["Dubai", "Abu Dhabi", "Sharjah"], size=n_samples),
    "plan": rng.choice(["basic", "premium"], size=n_samples),
})

risk_score = (
    (data["income"] < 4500).astype(int)
    + (data["age"] < 30).astype(int)
    + (data["plan"] == "basic").astype(int)
)

data["defaulted"] = (risk_score >= 2).astype(int)


# ------------------------------------------------------------
# 2. Separate X and y
# ------------------------------------------------------------

X = data.drop(columns="defaulted")
y = data["defaulted"]


# ------------------------------------------------------------
# 3. Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# 4. Define column groups
# ------------------------------------------------------------

numeric_features = ["age", "income"]
categorical_features = ["city", "plan"]


# ------------------------------------------------------------
# 5. Numeric preprocessing
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ------------------------------------------------------------
# 6. Categorical preprocessing
# ------------------------------------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ------------------------------------------------------------
# 7. Combine preprocessing
# ------------------------------------------------------------

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ------------------------------------------------------------
# 8. Build full pipeline
# ------------------------------------------------------------

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])


# ------------------------------------------------------------
# 9. Define hyperparameter grid
# ------------------------------------------------------------

param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [3, 5, 10, None],
    "clf__min_samples_split": [2, 5, 10],
}


# ------------------------------------------------------------
# 10. Define cross-validation strategy
# ------------------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 11. Build GridSearchCV
# ------------------------------------------------------------

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)


# ------------------------------------------------------------
# 12. Fit GridSearchCV on training data only
# ------------------------------------------------------------

grid_search.fit(X_train, y_train)


# ------------------------------------------------------------
# 13. Show best results from cross-validation
# ------------------------------------------------------------

print("Best parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1 score:")
print(grid_search.best_score_)


# ------------------------------------------------------------
# 14. Evaluate best model on untouched test data
# ------------------------------------------------------------

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nTest F1 score:")
print(f1_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))