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
train = torch.tensor(train.values)
lables = torch.tensor(lables.values)
test = torch.tensor(test.values)
train = data.TensorDataset(train, lables)
test = data.TensorDataset(test)
train_dataiter = data.DataLoader(train, batch_size=210, shuffle=True, num_workers=4)
test_dataiter = data.DataLoader(test, batch_size=280, shuffle=False, num_workers=4)

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        # self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.relu(self.fc3(x))
        # x = self.softmax(self.fc3(x))
        return x


# Train model
def train_model(model, train_dataiter, num_epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(train_dataiter):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch: %d, Iter: %d, Loss: %.4f' % (epoch, i, loss.item()))


def predict(model, test_dataiter):
    model.eval()
    model.to(device)
    result = []
    for i, data in enumerate(test_dataiter):
        inputs = data[0]
        inputs = inputs.to(device)
        outputs = model(inputs.float())
        outputs = torch.softmax(outputs, dim=1)
        result.append(outputs)
    return result


if __name__ == '__main__':
    model = Net()
    train_model(model, train_dataiter, 10)
    result = predict(model, test_dataiter)
    result = torch.cat(result, dim=0)
    result = torch.argmax(result, dim=1)
    result = result.cpu().numpy()
    submission = pd.DataFrame({'ImageId': range(1, 28001), 'Label': result})
    submission.to_csv('./Kaggle/HandNum/data/submission.csv', index=False)