import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv('data/climate.csv')
# data = data.drop(columns='Economic_Impact_Million_USD')
# data = data[data['Country'] == 'China']
# data = data.drop(columns=['Country'])
# data = data.sample(frac=1).reset_index(drop=True)
labels = data.pop('Crop_Yield_MT_per_HA')
data = data[['Crop_Type', 'Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT',
             'Irrigation_Access_%', 'Pesticide_Use_KG_per_HA', 'Economic_Impact_Million_USD']]
numeric_cols = data.select_dtypes(exclude='object').columns
data = pd.get_dummies(data, columns=['Crop_Type'], dummy_na=True)
# 对数值特征归一化
# data = (data - data.mean()) / data.std()
data[numeric_cols] = data[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
# data = pd.get_dummies(data, dummy_na=True)
data = data * 1
temp_data = data.copy()
num_train = int(data.shape[0] * 0.9)
train_data = data.iloc[:num_train]
test_data = data.iloc[num_train:]
train_labels = labels.iloc[:num_train]
test_labels = labels.iloc[num_train:]
data = torch.tensor(data.values, dtype=torch.float32, device=device)
labels = torch.tensor(labels.values, dtype=torch.float32, device=device)
train_data = torch.tensor(train_data.values, dtype=torch.float32, device=device)
test_data = torch.tensor(test_data.values, dtype=torch.float32, device=device)
train_labels = torch.tensor(train_labels.values, dtype=torch.float32, device=device)
test_labels = torch.tensor(test_labels.values, dtype=torch.float32, device=device)
train_labels = train_labels.view(-1, 1)
test_labels = test_labels.view(-1, 1)
labels = labels.view(-1, 1)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(17, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, X):
        # return self.net(X)
        X = self.fc1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.fc2(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.fc3(X)
        return X


def train():
    net = MLP()
    net = net.to(device)
    loss = nn.MSELoss()
    num_epochs, lr = 400, 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_hat = net(train_data)
        l = loss(y_hat, train_labels)
        l.backward()
        optimizer.step()
        print(f'epoch {epoch + 1}, loss {l.item()}')
    net.eval()

    def r2_score(y_true, y_pred):
        y_true_mean = torch.mean(y_true)
        ss_total = torch.sum((y_true - y_true_mean) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2

    y_pred = net(test_data)
    feature_importance = net.fc1.weight.data.cpu().numpy().mean(axis=0)
    # 获取特征名称
    feature_names = temp_data.columns
    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    indices = np.argsort(feature_importance)
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))  # 使用颜色映射
    plt.barh(range(len(indices)), feature_importance[indices], align='center', color=colors)
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    # plt.plot(test_labels.cpu().numpy(), label='actual')
    # plt.plot(y_pred.detach().cpu().numpy(), label='predicted')
    # plt.legend()
    plt.show()

    print('test loss:', loss(y_pred, test_labels))
    print('r2 score:', r2_score(test_labels, y_pred))


train()
