import pandas as pd

df = pd.read_csv("Liga1Proce.csv")
print(df["Resultado"].value_counts())
