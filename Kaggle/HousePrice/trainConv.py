import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

device = torch.device('mps')

# Load data
train = pd.read_csv('./Kaggle/HousePrice/data/train.csv')
test = pd.read_csv('./Kaggle/HousePrice/data/test.csv')
# train = pd.read_csv('./data/train.csv')
# test = pd.read_csv('./data/test.csv')

# Preprocess data
label = train['SalePrice']
train = train.drop(['Id', 'SalePrice'], axis=1)
test = test.drop(['Id'], axis=1)
n_train = train.shape[0]
n_test = test.shape[0]
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
        # 330 features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(1, 0))
        # 32 * 330 features
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        # 32 * 165 features
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        # 64 * 165 features
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        # 64 * 82 features
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        # 128 * 82 features
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        # 128 * 41 features
        self.fc1 = nn.Linear(128 * 41, 2048)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.unsqueeze(3)
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 128 * 41)
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
# Train model
def train_model(model, data_iter, num_epochs=100):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y.view(-1, 1))
            loss.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, loss.item()))


def predict(model, data_iter):
    model = model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, _ in data_iter:
            X = X.to(device)
            pred = model(X)
            predictions.append(pred)
    predictions = torch.cat(predictions, dim=0)
    return predictions.cpu().detach().numpy()


if __name__ == '__main__':
    net = Net()
    train_data = torch.tensor(train.values, dtype=torch.float)
    train_label = torch.tensor(label.values, dtype=torch.float)
    train_set = torch.utils.data.TensorDataset(train_data, train_label)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=n_train, shuffle=True)
    train_model(net, train_iter, 500)
    test_data = torch.tensor(test.values, dtype=torch.float)
    test_set = torch.utils.data.TensorDataset(test_data, torch.zeros(test_data.shape[0]))
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=n_test, shuffle=False)
    pred = predict(net, test_iter)
    pred = pred * label_std + label_mean
    submission = pd.DataFrame({'Id': range(1461, 2920), 'SalePrice': pred.flatten()})
    submission.to_csv('./Kaggle/HousePrice/data/submission.csv', index=False)
