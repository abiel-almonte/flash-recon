import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Softplus(),
        )

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8 * 8 * 9, kernel_size=(1, 1), padding=(0, 0))
        )

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch * num, ch, ht, wd)

        _, ix = torch.unique(ii, sorted=True, return_inverse=True)
        net = self.relu(self.conv1(net))
        net = net.view(batch, num, 128, ht, wd)

        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))
        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8 * 8 * 9, ht, wd)

        return 0.01 * eta, upmask
