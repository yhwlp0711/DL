import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Preprocess data
label = train['Survived']
train = train.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
n_train = train.shape[0]
all_data = pd.concat([train, test], axis=0)
numeric_cols = all_data.select_dtypes(include='number').columns
all_data[numeric_cols] = all_data[numeric_cols].apply(lambda x: x.fillna(x.mean().round().astype(int)), axis=0)
all_data = pd.get_dummies(all_data, dummy_na=True)
all_data = all_data * 1
train = all_data.iloc[:n_train]
test = all_data.iloc[n_train:]


# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Train model
def train_model(model, train, label, epochs=1000, lr=0.01):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train, label = train.to(device), label.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: loss {loss.item()}')


def predict(model, test):
    model.to(device)
    test = test.to(device)
    with torch.no_grad():
        output = model(test)
        _, predicted = torch.max(output, 1)
    return predicted.cpu().numpy()


if __name__ == '__main__':
    model = Net()
    train_tensor = torch.tensor(train.values, dtype=torch.float32)
    label_tensor = torch.tensor(label.values, dtype=torch.long)
    test_tensor = torch.tensor(test.values, dtype=torch.float32)
    train_model(model, train_tensor, label_tensor)
    prediction = predict(model, test_tensor)
    submission = pd.DataFrame({'PassengerId': test.index+892, 'Survived': prediction})
    submission.to_csv('./data/submission.csv', index=False)
