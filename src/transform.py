# Assisted by ChatGPT

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load data
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

# Separate features & target
X_train = train.drop("price", axis=1)
y_train = train["price"]

X_test = test.drop("price", axis=1)
y_test = test["price"]

# Keep only numeric columns
X_train = X_train.select_dtypes(include=['int64', 'float64'])
X_test = X_test.select_dtypes(include=['int64', 'float64'])

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save transformed data
os.makedirs("data/transformed", exist_ok=True)

pd.DataFrame(X_train_scaled).to_csv("data/transformed/X_train.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("data/transformed/X_test.csv", index=False)
y_train.to_csv("data/transformed/y_train.csv", index=False)
y_test.to_csv("data/transformed/y_test.csv", index=False)

print("Data transformed successfully")