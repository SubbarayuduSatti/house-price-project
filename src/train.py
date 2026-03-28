# Assisted by ChatGPT

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# load data
data = pd.read_csv("../data/data.csv")

print("Data Preview:")
print(data.head())

# select features
X = data[['sqft_living', 'bedrooms', 'bathrooms']]
y = data['price']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# -------------------------
# Model 1: Linear Regression
# -------------------------
model1 = LinearRegression()
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)
error1 = mean_absolute_error(y_test, pred1)

print("\nLinear Regression Error:", error1)

# -------------------------
# Model 2: Decision Tree
# -------------------------
model2 = DecisionTreeRegressor()
model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)
error2 = mean_absolute_error(y_test, pred2)

print("Decision Tree Error:", error2)