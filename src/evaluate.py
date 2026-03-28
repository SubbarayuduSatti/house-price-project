# Assisted by ChatGPT

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import json
import numpy as np

# Load data
X_test = pd.read_csv("data/transformed/X_test.csv")
y_test = pd.read_csv("data/transformed/y_test.csv")

# Load models
lr = joblib.load("models/linear_regression.pkl")
dt = joblib.load("models/decision_tree.pkl")

# Predictions
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)

# Errors

lr_error = np.sqrt(mean_squared_error(y_test, lr_pred))
dt_error = np.sqrt(mean_squared_error(y_test, dt_pred))

# Save metrics
metrics = {
    "LinearRegression_RMSE": lr_error,
    "DecisionTree_RMSE": dt_error
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(metrics)