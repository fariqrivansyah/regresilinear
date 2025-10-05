# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

# load data bersih
df = pd.read_csv("advertising.csv")  # atau "Advertising.csv" jika tak pakai IQR

# variabel
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
model = LinearRegression()
model.fit(X_train, y_train)

# prediksi
y_pred = model.predict(X_test)

# evaluasi
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

# interpretasi sederhana koefisien
coefs = dict(zip(X.columns, model.coef_))
print("\nIntercept:", model.intercept_)
print("Koefisien:")
for k,v in coefs.items():
    print(f"  {k}: {v:.4f}")

# simpan model
joblib.dump(model, "advertising_lr_model.joblib")
print("Model disimpan ke advertising_lr_model.joblib")
