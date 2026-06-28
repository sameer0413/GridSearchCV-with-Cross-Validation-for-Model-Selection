## GridSearchCV with Cross-Validation for Model Selection

This project uses GridSearchCV with cross-validation to select the best model hyperparameters.

Hyperparameters are settings chosen before training.

Examples:

- n_estimators
- max_depth
- min_samples_split

These settings control how the model learns.

### Why GridSearchCV Is Used

GridSearchCV tests multiple hyperparameter combinations and compares them using cross-validation.

Example:

n_estimators = 50, 100, 200

max_depth = 3, 5, 10, None

min_samples_split = 2, 5, 10

GridSearchCV tries every possible combination.

The combination with the best average cross-validation score is selected.

### Cross-Validation

Cross-validation splits the training data into multiple folds.

In 5-fold cross-validation:

- the model trains on 4 folds
- the model validates on 1 fold
- this repeats 5 times

The final cross-validation score is the average score across the folds.

This gives a more reliable estimate than using only one validation split.

### StratifiedKFold

StratifiedKFold is used for classification problems.

It keeps the class distribution similar in each fold.

This is important when the target classes are imbalanced.

Example:

If the full dataset has 80% class 0 and 20% class 1, each fold should have a similar ratio.

### Parameter Grid

The parameter grid defines which hyperparameters should be tested.

Example:

param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [3, 5, 10, None],
    "clf__min_samples_split": [2, 5, 10]
}

The double underscore is used to access parameters inside a Pipeline step.

Example:

clf__n_estimators

means:

change n_estimators inside the Pipeline step named clf.

### Training Data Only

GridSearchCV should be fitted only on the training data.

Example:

grid_search.fit(X_train, y_train)

The test data should not be used during hyperparameter tuning.

Using the test data during tuning causes data leakage.

Data leakage makes model performance look better than it really is.

### Final Test Evaluation

After GridSearchCV finds the best hyperparameters, the best model is evaluated on the test set.

Example:

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

The test score is the final honest estimate of model performance.

### Why This Workflow Works

This workflow separates model selection from final evaluation.

The training data is used for:

- fitting the model
- cross-validation
- hyperparameter tuning

The test data is used only once at the end.

This makes the evaluation more trustworthy and helps prevent overfitting to the test set.