import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

device = torch.device('mps')

# Load data
train = pd.read_csv("./Kaggle/Loan/data/train.csv")
test = pd.read_csv("./Kaggle/Loan/data/test.csv")
labels = train['loan_status']
train = train.drop(['id', 'loan_status'], axis=1)
num_train = train.shape[0]
test = test.drop(['id'], axis=1)
all_data = pd.concat([train, test], axis=0)
numeric_cols = all_data.select_dtypes(exclude='object').columns
all_data[numeric_cols] = all_data[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
all_data = pd.get_dummies(all_data, dummy_na=True)
train = all_data.iloc[:num_train]
test = all_data.iloc[num_train:]

# Define model