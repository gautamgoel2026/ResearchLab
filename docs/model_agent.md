# Model Agent

Owner: Nina Saravanan

The model agent takes the cleaned dataset from the data agent, trains baseline
classifiers, and returns metrics that the visualization agent can chart.

## Baselines

| Model | Why |
|---|---|
| Logistic regression | Simple, fast, easy to explain. Sets the floor every other model must beat. |
| Random forest | Handles non-linear patterns and mixed feature types with little tuning. Gives feature importances directly. |

Default: train both and report both. The best model (by F1) is the one that gets
saved and is used for feature importances.

Settings:
- Train/test split: 80/20, stratified on the target, `random_state=42`.
- Logistic regression: `max_iter=1000`, features scaled with `StandardScaler`.
- Random forest: `n_estimators=200`, `random_state=42`.

## Metrics

| Metric | Why |
|---|---|
| Accuracy | Easy to understand, but misleading when classes are unbalanced (e.g. churn). |
| F1 | Balances precision and recall, so it is the main metric for picking the best model. Uses `average="binary"` for two classes and `average="weighted"` otherwise. |
| Confusion matrix | Shows which kinds of mistakes the model makes. Used for D's confusion matrix chart. |

All metrics are computed on the test set only.

## Input

```python
train_model(data_path: str, target: str, model: str = "both") -> dict
```

- `data_path`: path to the cleaned CSV from the data agent, normally
  `results/cleaned.csv`.
- `target`: name of the column to predict.
- `model`: `"logistic_regression"`, `"random_forest"`, or `"both"` (default).

Expected from the data agent's output:
- No missing values.
- All feature columns numeric (categoricals already encoded).
- Target column present, with no leakage columns left in.

## Output

`train_model` returns a dictionary:

```python
{
    "target": "Survived",
    "n_train": 712,
    "n_test": 179,
    "class_labels": [0, 1],
    "best_model": "random_forest",
    "models": {
        "logistic_regression": {
            "accuracy": 0.80,
            "f1": 0.74,
            "confusion_matrix": [[95, 15], [21, 48]]
        },
        "random_forest": {
            "accuracy": 0.82,
            "f1": 0.76,
            "confusion_matrix": [[98, 12], [20, 49]]
        }
    },
    "feature_importances": {"Sex": 0.27, "Fare": 0.25, "Age": 0.24},
    "model_path": "results/model.pkl",
    "metrics_path": "results/metrics.json"
}
```

Notes for the visualization agent:
- `confusion_matrix` is a plain list of lists. Rows are the true class and
  columns are the predicted class, both in the order of `class_labels`.
- `feature_importances` is sorted from most to least important and comes from
  `best_model`. For logistic regression it uses the absolute coefficients.
- Numbers are rounded to 4 decimal places in the real output.

## Files saved to `results/`

| File | Contents |
|---|---|
| `results/model.pkl` | The best trained model, saved with `joblib`. |
| `results/metrics.json` | The same dictionary that `train_model` returns. |

## Errors

`train_model` raises a `ValueError` with a clear message when:
- the file at `data_path` does not exist,
- `target` is not a column in the data,
- the data still has missing values or non-numeric feature columns,
- the target has only one class.

The orchestrator catches these, retries once, then reports the error.
