import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

#set device
device = torch.device('mps')

# Load the data
train_data = pd.read_csv('./Kaggle/TitanicSpace/data/train.csv')
test_data = pd.read_csv('./Kaggle/TitanicSpace/data/test.csv')

# Preprocess the data
label = train_data['Transported']
train_data = train_data.drop(['Name', 'Transported'], axis=1)
test_Id = test_data['PassengerId']
test_data = test_data.drop(['Name'], axis=1)
num_train = train_data.shape[0]
all_data = pd.concat([train_data, test_data], axis=0)
all_data['PassengerId'] = all_data['PassengerId'].apply(lambda x: float(x.split('_')[0]))
all_data['Cabin'] = all_data['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 'U')
numeric_cols = all_data.select_dtypes(include='number').columns
all_data[numeric_cols] = all_data[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=0)
all_data[numeric_cols] = all_data[numeric_cols].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
all_data = pd.get_dummies(all_data, dummy_na=True)
all_data = all_data * 1
train_data = all_data.iloc[:num_train]
test_data = all_data.iloc[num_train:]

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(25, 256)
        self.fc2 = nn.Linear(256, 128)
        # elf.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 2)
        # self.fc4 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc3(x))
        x = self.fc3(x)
        # x = self.dropout(x)
        # x = self.fc4(x)
        return x
    

def train_model(model, train, label, epochs=700, lr=0.01):
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
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


def predict(model, test):
    model.to(device)
    test = test.to(device)
    with torch.no_grad():
        output = model(test)
        _, predicted = torch.max(output, 1)
        return predicted.cpu().numpy()


# Train the model
if __name__ == '__main__':
    net = Net()
    train_model(net, torch.tensor(train_data.values).float(), torch.tensor(label.values))
    predicted = predict(net, torch.tensor(test_data.values).float())
    transported = predicted == 1
    submission = pd.DataFrame({'PassengerId': test_Id, 'Transported': transported})
    submission.to_csv('./Kaggle/TitanicSpace/data/submission.csv', index=False)