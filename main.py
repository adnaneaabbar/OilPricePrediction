import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import load_model

df = pd.read_csv("data/DCOILBRENTEU.csv", sep=",")
print(df.shape)

print(df.head())

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

train_size = get_train_size(df, batch_size, test_percent) + timesteps * 2
df_train = df[0: train_size]
training_set = df_train.iloc[:, 1:2].values
print(training_set.shape)                                     # 7892
