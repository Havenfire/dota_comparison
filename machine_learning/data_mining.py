from sklearn.preprocessing import MinMaxScaler
from util import generate_training_data
import pandas as pd

# Read the CSV file
try:
    df = pd.read_csv("hero_data.csv")
except:
    generate_training_data()

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

identifier_column = 'heroId'
if identifier_column in numeric_columns:
    numeric_columns = numeric_columns.drop(identifier_column)

scaler = MinMaxScaler()

df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Save the normalized DataFrame back to the CSV file
df.to_csv("normalized_hero_data.csv", index=False)
