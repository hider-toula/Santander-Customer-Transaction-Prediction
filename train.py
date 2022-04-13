import torch
#from sklearn import metrics
import torch.nn as nn
import torch.optim as optim
from dataset import get_data
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NN(nn.Module):

    def __init__(self, input_size):
        super(NN, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)


model = NN(input_size=200)

train_ds, val_ds, test_ds, test_ids = get_data()
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1024)
val_loader = DataLoader(val_ds, batch_size=1024)

x, y = next(iter(train_loader))
print(x.shape)
