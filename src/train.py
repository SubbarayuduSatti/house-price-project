# Assisted by ChatGPT

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import joblib
import os

# Load transformed data
X_train = pd.read_csv("data/transformed/X_train.csv")
y_train = pd.read_csv("data/transformed/y_train.csv")

# Models
lr = LinearRegression()
dt = DecisionTreeRegressor()

# Train
lr.fit(X_train, y_train.values.ravel())
dt.fit(X_train, y_train.values.ravel())

# Save models
os.makedirs("models", exist_ok=True)

joblib.dump(lr, "models/linear_regression.pkl")
joblib.dump(dt, "models/decision_tree.pkl")

print("Models trained successfully")