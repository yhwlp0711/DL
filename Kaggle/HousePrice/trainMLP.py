import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load data
# train = pd.read_csv('./Kaggle/HousePrice/data/train.csv')
# test = pd.read_csv('./Kaggle/HousePrice/data/test.csv')
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Preprocess data
label = train['SalePrice']
train = train.drop(['Id', 'SalePrice'], axis=1)
test = test.drop(['Id'], axis=1)
n_train = train.shape[0]
all_data = pd.concat([train, test], axis=0)
numeric_cols = all_data.select_dtypes(include='number').columns
all_data[numeric_cols] = all_data[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=0)
all_data[numeric_cols] = all_data[numeric_cols].apply(lambda x: (x-x.mean())/x.std(), axis=0)
all_data = pd.get_dummies(all_data, dummy_na=True)
all_data = all_data * 1
# all_data = all_data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
label_mean = label.mean()
label_std = label.std()
label = (label - label_mean) / label_std
train = all_data.iloc[:n_train]
test = all_data.iloc[n_train:]

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(330, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 1024)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, x):
        # x = self.fc1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
    
# Train model
if __name__ == '__main__':
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(300):
        optimizer.zero_grad()
        output = net(torch.tensor(train.values, dtype=torch.float))
        loss = criterion(output, torch.tensor(label.values, dtype=torch.float).view(-1, 1))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        print('Epoch %d, loss: %.6f' % (epoch, loss.item()))

    # torch.save(net.state_dict(), './Kaggle/HousePrice/model.pth')
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(test.values, dtype=torch.float))
        pred = pred * label_std + label_mean

    submission = pd.DataFrame({'Id': range(1461, 2920), 'SalePrice': pred.numpy().flatten()})
    submission.to_csv('./Kaggle/HousePrice/data/submission.csv', index=False)