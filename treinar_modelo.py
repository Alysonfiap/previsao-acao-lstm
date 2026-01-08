import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import joblib

# ===============================
# Coleta de Dados
# ===============================
symbol = "DIS"
start_date = datetime(2018, 1, 1)
end_date = datetime(2024, 7, 20)

df = web.DataReader(symbol, data_source="stooq", start=start_date, end=end_date)
df = df.sort_index()

# ===============================
# Usar somente Close
# ===============================
data = df[['Close']].values

# ===============================
# Normalização
# ===============================
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

# ===============================
# Criar sequências
# ===============================
def create_sequences(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(data_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ===============================
# Treino / Teste
# ===============================
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ===============================
# Modelo
# ===============================
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")

print("Treinando...")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# ===============================
# Avaliação
# ===============================
test_pred = model.predict(X_test)

test_pred_real = scaler.inverse_transform(test_pred)
y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))

rmse = math.sqrt(mean_squared_error(y_test_real, test_pred_real))
mae = mean_absolute_error(y_test_real, test_pred_real)
mape = np.mean(np.abs((y_test_real - test_pred_real) / y_test_real)) * 100
r2 = r2_score(y_test_real, test_pred_real)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f'Acurácia aproximada: {100-mape:.2f}%')
print(f"R²: {r2:.4f}")

# ===============================
# Salvar modelo + scaler
# ===============================
model.save("modelo_lstm_acao.h5")
joblib.dump(scaler, "scaler.pkl")

print("Modelo e scaler salvos com sucesso!")
