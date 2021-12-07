import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/DCOILBRENTEU.csv", sep=",")
print(df.shape)

print(df.head())

#remove NA values
df = df[df["DCOILBRENTEU"] != "."]
print(df.shape)

df_ind = df.set_index("DATE")

# Visualising the Data
df_ind = df_ind.astype(float)
df_ind.plot()
plt.show()
plt.savefig("data/initial_data.png")