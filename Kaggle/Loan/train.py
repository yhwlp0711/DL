import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import WeightedRandomSampler
from torch.nn import functional as F
import pandas as pd


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# device = torch.device('mps')
device = torch.device('cuda')

# Load data
# train = pd.read_csv("./Kaggle/Loan/data/train.csv")
# test = pd.read_csv("./Kaggle/Loan/data/test.csv")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
labels = train['loan_status']
train = train.drop(['id', 'loan_status'], axis=1)
num_train = train.shape[0]
num_test = test.shape[0]
ids = test['id']
test = test.drop(['id'], axis=1)
all_data = pd.concat([train, test], axis=0)
numeric_cols = all_data.select_dtypes(exclude='object').columns
all_data[numeric_cols] = all_data[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
all_data = pd.get_dummies(all_data)
all_data = all_data * 1
train = all_data.iloc[:num_train]
test = all_data.iloc[num_train:]
train = torch.Tensor(train.values)
labels = torch.Tensor(labels.values)
# train = TensorDataset(train, labels)
test = torch.Tensor(test.values)
class_sample_count = torch.tensor([(labels == t).sum() for t in torch.unique(labels, sorted=True)])
weight = 1. / class_sample_count.float()
labels_long = labels.to(torch.long)
samples_weight = torch.tensor([weight[t] for t in labels_long], dtype=torch.float)

# 创建 WeightedRandomSampler
sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

# 创建 TensorDataset 和 DataLoader
train_dataset = TensorDataset(train, labels)
train_dataiter = DataLoader(train_dataset, batch_size=num_train, sampler=sampler, num_workers=4)
# train_dataiter = DataLoader(train, batch_size=2048, shuffle=True, num_workers=4)
test_dataiter = DataLoader(test, batch_size=num_test, shuffle=False, num_workers=4)


# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(26, 128)
        self.fc2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        # x = self.sigmoid(x)
        return x


# Define training
def train(model, train_dataiter, num_epochs=100):
    model.to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(num_epochs):
        model.train()
        loss = 0
        for i, data in enumerate(train_dataiter):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            loss += loss.item()
        # scheduler.step()
        print('Epoch:', epoch, 'Loss:', loss)


# Define testing
def test(model, test_dataiter):
    model.to(device)
    model.eval()
    predictions = []
    for i, data in enumerate(test_dataiter):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs.float())
        outputs = torch.softmax(outputs, 1)
        predicted = outputs[:, 1]
        # _, predicted = torch.max(outputs, 1)
        predictions.append(predicted)
    return torch.cat(predictions).cpu().detach().numpy()


# Train model
if __name__ == '__main__':
    model = Net()
    train(model, train_dataiter, num_epochs=500)
    predictions = test(model, test_dataiter)
    submission = pd.DataFrame({'id': ids, 'loan_status': predictions})
    submission.to_csv('./data/submission.csv', index=False)
