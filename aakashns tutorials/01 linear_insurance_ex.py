'''
Building a pytorch based logistic regression model from scratch
Rahul Goel 
12/30/2021

Reference URL: https://jovian.ai/aakashns/02-insurance-linear-regression

'''
import pandas as pd
from torchvision.datasets.utils import download_url

# Download the dataset
'''
DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
DATA_FILENAME = "insurance.csv"
download_url(DATASET_URL, '.')
'''

# Section 1
print("== Section 1: EDA ==")

# Check the dataset
df = pd.read_csv('insurance.csv')
print(df.head())

print(f"Rows: {len(df)} | Columns: {len(df.columns)}") # 1338 x 6 features + 1 target

print(f"\n\nRange for target variable: {df['charges'].describe()}")
print("\n---------------------------------\n")

# Section 2
print("== Section 2: Prepare dataset ==")

# 1. Treat categorical variables
# 2. Split out features and labels
# 3. Convert to tensors of approrpriate size
# 4. Convert to training and testing batches using DataLoader

numeric_cols = df._get_numeric_data().columns 

y = df['charges']
X_reqd_columns = df.columns[0:len(df.columns)-1]
X = df[X_reqd_columns]

print("y", y) # 1338 x 1
print("X", X) # 1338 x 6 

