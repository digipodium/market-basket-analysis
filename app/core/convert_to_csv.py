import pandas as pd

df = pd.read_excel("online_retail.xlsx")
df.to_csv("online_retail.csv",index=False)
print("done")