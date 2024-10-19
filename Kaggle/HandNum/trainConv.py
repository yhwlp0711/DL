import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pandas as pd

device = torch.device('mps')

# Load data
train = pd.read_csv('./Kaggle/HandNum/data/train.csv')
test = pd.read_csv('./Kaggle/HandNum/data/test.csv')
lables = train['label']
train = train.drop('label', axis=1)
train = torch.tensor(train.values).reshape(-1, 1, 28, 28)
lables = torch.tensor(lables.values)
train = data.TensorDataset(train, lables)
test = torch.tensor(test.values).reshape(-1, 1, 28, 28)
test = data.TensorDataset(test)
train_dataiter = data.DataLoader(train, batch_size=210, shuffle=True, num_workers=4)
test_dataiter = data.DataLoader(test, batch_size=280, shuffle=False, num_workers=4)

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Train model
def train_model(model, train_dataiter, num_epochs=100):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        loss = 0
        for i, data in enumerate(train_dataiter):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss += loss.item()
        print('Epoch:', epoch, 'Loss:', loss)


# Test model
def test_model(model, test_dataiter):
    model.to(device)
    model.eval()
    result = []
    for i, data in enumerate(test_dataiter):
        inputs = data[0]
        inputs = inputs.to(device)
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs, 1)
        result.append(predicted)
    return torch.cat(result).cpu().numpy()
    

if __name__ == '__main__':
    model = Net()
    train_model(model, train_dataiter)
    predicted = test_model(model, test_dataiter)
    # predicted = predicted.cpu().numpy()
    data = pd.DataFrame({'ImageId': range(1, 28001), 'Label': predicted})
    data.to_csv('./Kaggle/HandNum/data/submission.csv', index=False)