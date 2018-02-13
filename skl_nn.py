from sknn.mlp import Regressor, Layer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# import data
print("💾 Start importing data")
df = pd.read_csv("gasoline.csv")
print("✅ Done importing data")

# define features and label
print("✂️ Divide and split data")
X = pd.DataFrame.as_matrix(df, columns=['date', 'month', 'year',
                                        'weekday', 'marke', 'latitude',
                                        'longitude', 'dautobahn',
                                        'autobahn', 'aral', 'esso',
                                        'jet', 'shell', 'total',
                                        'rotterdam', 'brent', 'wti',
                                        'eurusd', 'vehicles'])
y = pd.DataFrame.as_matrix(df, columns=['e5gas'])

# split into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("🥅 Fit a neural net or whatever")
nn = Regressor(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Linear")],
    learning_rate=0.02,
    n_iter=10)
nn.fit(X_train, y_train)
