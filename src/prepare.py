# Assisted by ChatGPT

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load data
data = pd.read_csv("C:/Users/91965/Desktop/house-price-project/data/data.csv")

# Split data (90/10)
train, test = train_test_split(data, test_size=0.1, random_state=42)

# Save
os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Data prepared successfully")