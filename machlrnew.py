import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Запрос тикера у пользователя
ticker = input("Введите тикер компании (например, AAPL, MSFT, GOOG): ")

# Загрузка данных
try:
    data = yf.download(ticker, period="1y")
    data['Close'] = data['Adj Close']
    data = data.dropna()
except Exception as e:
    print(f"Ошибка при загрузке данных для тикера {ticker}: {e}")
    exit()

# Подготовка данных
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Параметры модели
look_back = 30
forecast_days = 3  # Количество дней, на которые делаем прогноз

# Создание обучающих данных для многошагового прогноза
X, y = [], []
for i in range(look_back, len(scaled_data) - forecast_days):
    X.append(scaled_data[i - look_back:i, 0])
    y.append(scaled_data[i:i + forecast_days, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Создание и обучение LSTM модели
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(forecast_days))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=16)

# Прогнозирование
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Оценка модели
rmse_values = []
mape_values = []

for day in range(forecast_days):
    y_true = data['Close'].values[-(len(predictions) * forecast_days):]  # Get values for whole predicted interval
    y_pred = predictions[:, day]

    # Slice y_true for the current day of the forecast
    y_true_day = y_true[day::forecast_days]
    # Limit the true values to the length of the predictions
    y_true_day = y_true_day[:len(y_pred)]

    rmse = np.sqrt(mean_squared_error(y_true_day, y_pred))
    mape = mean_absolute_percentage_error(y_true_day, y_pred)
    rmse_values.append(rmse)
    mape_values.append(mape)
    print(f"День {day + 1} - RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

# Вывод последних прогнозов
print(f"Прогнозы цены закрытия на {forecast_days} дней вперёд с анализом прошлых {look_back} дней:")
for pred in predictions[-1:]:
    print(pred)