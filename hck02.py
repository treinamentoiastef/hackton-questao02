import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import style


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
style.use('ggplot')
plt.figure(figsize = (18,9))
plt.title('Fechamento X Data Dinotrade')
plt.plot(range(df.shape[0]),df["Close"],  linewidth=0.3)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500])
plt.xlabel('Data',fontsize=18)
plt.ylabel('Preço no Fechamento',fontsize=18)
# plt.show()

# Separa valor do fechamento
df_close = df[["Close"]]

# Faz escalonamento dos valores de fechamento
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_set = scaler.fit_transform(df_close)

# faz a divisão do set para treino e teste
train_set_size = int(len(scaled_set)*0.7)
train_set = scaled_set[0:train_set_size, :]
test_set_size = len(scaled_set) - train_set_size
test_set = scaled_set[train_set_size:, :]
# verifica se a separação foi bem sucedida
assert(len(scaled_set) == len(test_set) + len(train_set))
