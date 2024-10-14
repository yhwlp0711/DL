import os
import pandas as pd
import torch

# os.makedirs(os.path.join('.', 'data'), exist_ok=True)
dataFile = os.path.join('..', 'data', 'data.csv')
# with open(dataFile, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')
#     f.write('NA,Pave,127500\n')
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
# print('Data file created at: ', dataFile)

data = pd.read_csv(dataFile)
print(data)
inputs, outputs = data.iloc[:, :-1], data.iloc[:, -1]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)
print(outputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

inputs = inputs.astype('float32')
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)
