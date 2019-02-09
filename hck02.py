import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import time
import urllib.request, json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import style

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential


# import do dataset
df = pd.read_csv("dataset/trade.csv")
# print(df.head(3))

# fazendo parse das datas, excluindo "A. M." da string
df["Date"] = df["Date"].str.replace(r'A.M.', '')
df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y', errors='ignore')
# print(df["Date"][:3])

# Ordenando elementos pela data
# df = df.sort_values('Date')
# print(df.head(50))

# Visualização dos Dados "Puros"
# style.use('ggplot')
# plt.figure(figsize = (18,9))
# plt.title('Fechamento X Data Dinotrade')
# plt.plot(range(df.shape[0]),df["Close"],  linewidth=0.3)
# plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500])
# plt.xlabel('Data',fontsize=18)
# plt.ylabel('Preço no Fechamento',fontsize=18)
# plt.show()

# Separa os valores de abertura para feature e do fechamento para label
df_features = df[["Open"]]
df_label = df[["Close"]]

# Faz escalonamento dos valores de fechamento
scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled_set = scaler.fit_transform(df_features)
y_scaled_set = scaler.fit_transform(df_label)

# faz a divisão do set para treino e teste
train_set_size = int(len(x_scaled_set)*0.7)
test_set_size = len(x_scaled_set) - train_set_size

X_train = x_scaled_set[0:train_set_size, :]
y_train = y_scaled_set[0:train_set_size, :]
X_test = x_scaled_set[train_set_size:, :]
y_test = y_scaled_set[train_set_size:, :]


# verifica se a separação foi bem sucedida
assert(len(y_scaled_set) == len(X_test) + len(y_train))


model = Sequential()

model.add(LSTM(
    input_shape = (X_train.shape[1], 1),
    output_dim = 100,
    return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(
    units = 50,
    return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(
    units = 50,
    return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(
    units = 50))
model.add(Dropout(0.2))


model.add(Dense(
    units=1))
model.add(Activation('linear'))

start = time.time()
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model.fit(
    X_train,
    y_train,
    batch_size=128,
    nb_epoch=100,
    validation_split=0.05)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

score = model.evaluate(x=X_test, y=y_test, batch_size=128, verbose=1)

nx = np.reshape(x_scaled_set, (x_scaled_set.shape[0], x_scaled_set.shape[1], 1))

predicted = model.predict(nx)

results = scaler.inverse_transform(predicted)


# print(score*100)
style.use('ggplot')
plt.figure(figsize = (18,9))
plt.title('Fechamento X Data Dinotrade')
plt.plot(range(df.shape[0]),df["Close"],  linewidth=0.3, color="r")
plt.plot(range(df.shape[0]),results,  linewidth=0.3,color="b")
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500])
plt.xlabel('Data',fontsize=18)
plt.ylabel('Preço no Fechamento',fontsize=18)

plt.show()
