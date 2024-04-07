import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from numpy import *
from math import sqrt
from pandas import *
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM, LeakyReLU
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot
from pickle import load

X_train = np.load("X_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
X_test = np.load("X_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)
yc_train = np.load("yc_train.npy", allow_pickle=True)
yc_test = np.load("yc_test.npy", allow_pickle=True)

# Parameters
LR = 0.001
BATCH_SIZE = 64
N_EPOCH = 200

input_dim = X_train.shape[1]
feature_size = X_train.shape[2]
output_dim = y_train.shape[1]


def build_lstm_model(input_dim, feature_size, output_dim):
    model = Sequential()
    model.add(LSTM(units=500, input_shape=(input_dim, feature_size), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=500, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=220, activation='relu'))
    model.add(Dense(units=output_dim))

    optimizer = Adam(lr=LR)
    model.compile(optimizer=optimizer, loss='mse')

    return model


def out_of_sample_r_squared(actual_returns, predicted_returns):
    """
    Calculate the out-of-sample R^2 for stock return predictions.

    Parameters:
    - actual_returns: actual monthly returns of stocks.
    - predicted_returns: predicted monthly returns of stocks.

    Returns:
    - R_os^2: Out-of-sample R squared.
    """
    numerator = np.sum((actual_returns - predicted_returns) ** 2)
    denominator = np.sum(actual_returns ** 2)
    R_os_squared = 1 - (numerator / denominator)

    return R_os_squared


model = build_lstm_model(input_dim, feature_size, output_dim)


# Fit model
history = model.fit(X_train, y_train, epochs=N_EPOCH, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                    verbose=2, shuffle=False)

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

print(model.summary())

# Predict and evaluate
yhat = model.predict(X_test, verbose=0)
r2oos = out_of_sample_r_squared(y_test, yhat)
rmse = sqrt(mean_squared_error(y_test, yhat))
print('Test R2oos: %.3f' % r2oos)
print('Test RMSE: %.3f' % rmse)