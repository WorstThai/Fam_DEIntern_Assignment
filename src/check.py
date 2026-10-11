import pandas as pd

df = pd.read_csv("data/output_file.csv", parse_dates=["date"])

print(df.head())