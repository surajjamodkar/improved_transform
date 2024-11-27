import polars as pl
from large_transform.GMM import DataTransformer

# Initialize data 
train_data = pl.read_csv("Credit.csv")[["Amount", "V1"]]

# for increasing scale. change number to scale by double. i.e. 3 will make the data 2^3 long.
for _ in range(2):
    train_data = pl.concat([train_data,train_data])

print(f"Data Size - {train_data.shape[0]:_}, {train_data.shape[1]}")

transformer = DataTransformer(train_data=train_data)
# fit models using SparkML
transformer.fit()
# transform other data 
transformer.transform(train_data, save_parquet= True)
# inverse transform data 
transformer.inverse_transform(pl.read_parquet('transformed_values/transformed_values.parquet'))