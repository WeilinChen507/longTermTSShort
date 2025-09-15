import numpy as np
import pandas as pd


import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from utils.basic_model import MLP


class mlp_estimator(nn.Module):

    def __init__(self, x_dims, y_dims, n_layers=3, hidden_dim=50, device='cuda', activation='lrelu'):
        super().__init__()
        self.mlp = MLP(input_dim=x_dims, output_dim=y_dims, hidden_dim=hidden_dim, n_layers=n_layers,
                       activation=activation, device=device)

    def forward(self, x):
        return self.mlp(x)

    def fit(self, x, y, epoch=100, batch_size=128, lr=1e-3, weight_decay=1e-4):
        x,y = torch.tensor(x).cuda(), torch.tensor(y).cuda()
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)


        for _ in range(epoch):
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                (x, y) = data
                # print(x.shape)
                # print(y.shape)
                outputs = self.forward(x)
                # print(outputs.shape)

                loss = torch.mean(torch.square(outputs - y))
                loss.backward()
                optimizer.step()

    def predict(self, x):
        '''
        :param x:  np.array
        :return:   np.array
        '''
        x = torch.tensor(x).cuda()
        y = self.forward(x)
        y = y.detach().cpu().numpy()
        return y


# Warnings configuration
import warnings
# warnings.filterwarnings('ignore')
if __name__ == '__main__':
    torch.set_default_dtype(torch.double)
    x = np.random.randn(1000)
    y = 3*x + np.random.randn(1000)

    x,y = x[:,None], y[:,None]
    print(x.shape)
    print(y.shape)

    model = mlp_estimator(1,1).cuda()
    model.fit(x,y)

    x = np.random.randn(1000)+1
    x = x[:,None]
    y = model.predict(x)
    print(np.mean(y))

