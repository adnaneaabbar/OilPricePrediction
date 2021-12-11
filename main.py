import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Input, LSTM
from keras.models import Model, load_model

df = pd.read_csv("data/DCOILBRENTEU.csv", sep=",")
# print(df.shape)                                             # 9009, 2

# print(df.head())

# remove NA values
df = df[df["DCOILBRENTEU"] != "."]
# print(df.shape)

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

# setting the test size
def get_test_size(data, batch_size):
    n = len(data)
    for i in range(n - batch_size, n - timesteps * 2):
        if (i - train_size) % batch_size == 0:
            return i


# print(len(df) * (1 - test_percent))                         # 7890.3
# print(get_train_size(df, batch_size, test_percent))         # 7872

train_size = get_train_size(df, batch_size, test_percent) + 2 * timesteps
df_train = df[0: train_size]  # 7892, 2
training_set = df_train.iloc[:, 1:2].values
# print(training_set.shape)                                   # 7892, 1

# print(len(df) * (test_percent))                             # 876.7
# print(get_test_size(df, batch_size))                        # 8724

test_size = get_test_size(df, batch_size) + 2 * timesteps
df_test = df[train_size: test_size]  # 7892, 2
test_set = df_test.iloc[:, 1:2].values
# print(test_set.shape)                                   # 852, 1

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(training_set)
test_scaled = scaler.fit_transform(test_set)
# print(train_scaled.shape)                                  # 7892, 1
# print(test_scaled.shape)                                   # 852, 1

# Constructing X, Y train
x_train = []
y_train = []
for i in range(timesteps, train_size - timesteps):
    x_train.append(train_scaled[i - timesteps:i, 0])
    y_train.append(train_scaled[i:i + timesteps, 0])

x_test = []
for i in range(timesteps, (test_size - train_size) - timesteps):
    x_test.append(test_scaled[i-timesteps:i, 0])

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
x_test = np.asarray(x_test).astype(np.float32)
# print(x_train.shape)  # 7872, 10
# print(y_train.shape)  # 7872, 10
# print(x_test.shape)  # 832, 10

# reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# print(x_train.shape)  # 7872, 10, 1
# print(y_train.shape)  # 7872, 10, 1
# print(x_test.shape)  # 832, 10

# model training
'''
inputs = Input(batch_shape=(batch_size, timesteps, 1))
lstm_1 = LSTM(10, stateful=True, return_sequences=True)(inputs)
lstm_2 = LSTM(10, stateful=True, return_sequences=True)(lstm_1)
output_1 = Dense(units=1)(lstm_2)
model = Model(inputs=inputs, outputs=output_1)
model.compile(optimizer='adam', loss='mae')
model.summary()

for i in range(epochs):
    print("Epoch: " + str(i))
    model.fit(x_train, y_train, shuffle=False, epochs=1, batch_size=batch_size)
    model.reset_states()

model.save(filepath="models/model_with_10ts.h5")
'''

model = load_model("models/model_with_10ts.h5")

# prediction
predicted_test = model.predict(x_test, batch_size=batch_size)
model.reset_states()
# print(predicted_test.shape)  # 832, 10, 1

# reshaping
predicted_test = np.reshape(predicted_test, (predicted_test.shape[0], predicted_test.shape[1]))
# print(predicted_test.shape)  # 832, 10

# inverse transform
predicted_test = scaler.inverse_transform(predicted_test)

# y_test
y_test = []
#(test_size - train_size) - timesteps
for i in range(0, (test_size - train_size) - timesteps * 2):
    y_test = np.append(y_test, predicted_test[i, timesteps-1])
# print(y_test.shape)   # 832

# reshaping
y_test = np.reshape(y_test, (y_test.shape[0], 1))
# print(y_test.shape)   # 832, 1