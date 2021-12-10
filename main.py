import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import load_model

df = pd.read_csv("data/DCOILBRENTEU.csv", sep=",")
# print(df.shape)                                             # 9009, 2

# print(df.head())

#remove NA values
df = df[df["DCOILBRENTEU"] != "."]
print(df.shape)

# df_ind = df.set_index("DATE")
#
# # Visualising the Data
# df_ind = df_ind.astype(float)
# df_ind.plot()
# plt.savefig("data/data.png")
# plt.show()

batch_size = 64
epochs = 120
timesteps = 10
test_percent = 0.1

# setting the training size
def get_train_size(data, batch_size, test_percent):
    n = len(data)
    n *= (1 - test_percent)
    if int(n) % batch_size == 0:
        return int(n)
    for i in range(int(n) - batch_size, int(n)):
        if i % batch_size == 0:
            return i

# print(len(df) * (1 - test_percent))                         # 7890.3
# # print(get_train_size(df, batch_size, test_percent))       # 7872

train_size = get_train_size(df, batch_size, test_percent) + 2 * timesteps
df_train = df[0: train_size]                                  # 7892, 2
training_set = df_train.iloc[:, 1:2].values
# print(training_set.shape)                                   # 7892, 1

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(training_set)
# print(train_scaled.shape)                                   # 7892, 1

# Constructing X, Y train
x_train = []
y_train = []
for i in range(timesteps, train_size - timesteps):
    x_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i:i+timesteps, 0])

x_train = np.array(x_train, dtype=object)
y_train = np.array(y_train, dtype=object)
print(x_train.shape)                                   # 7872, 10
print(y_train.shape)                                   # 7872, 10
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
print(x_train.shape)                                   # 7872, 10, 1
print(y_train.shape)                                   # 7872, 10, 1

inputs = Input(batch_shape=(batch_size, timesteps, 1))
lstm_1 = LSTM(10, stateful=True, return_sequences=True)(inputs)
lstm_2 = LSTM(10, stateful=True, return_sequences=True)(lstm_1)
output_1 = Dense(units=1)(lstm_2)
model = Model(inputs=inputs, outputs=output_1)
model.compile(optimizer='adam', loss='mae')
model.summary()