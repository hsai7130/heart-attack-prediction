import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    "age": [45, 50, 60, 35],
    "chol": [200, 240, 300, 180],
    "target": [0, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[["age", "chol"]]
y = df["target"]

model = LogisticRegression()
model.fit(X, y)

print(model.predict([[55, 250]]))
