import pandas as pd
import numpy as np

df = pd.read_csv("data/DCOILBRENTEU.csv", sep=",")
print(df.shape)

print(df.head())