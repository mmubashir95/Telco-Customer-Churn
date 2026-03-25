from src.config import DATA_PATH
from src.data.data_loader import load_raw_data

df = load_raw_data(DATA_PATH)

print(df.shape)
print(df.head())

