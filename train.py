import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# ダミーデータ (y = 2x + 1)
X = np.random.rand(100, 1) * 10  # 0～10のランダムな値
y = 2 * X + 1 + np.random.randn(100, 1)  # ノイズを加える

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルのトレーニング
model = LinearRegression()
model.fit(X_train, y_train)

# モデルの評価
predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))

# モデルの保存
joblib.dump(model, "linear_model.pkl")
print("Model saved as 'linear_model.pkl'")
