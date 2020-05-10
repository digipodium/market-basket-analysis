import pandas as pd

df = pd.read_excel("cleaned_online_retail.xlsx")
df.to_csv("cleaned_online_retail.csv",index=False)
print("done")