import torch
import torch.nn as nn

from .gru import ConvGRU
from .graph_agg import GraphAgg


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2 * 3 + 1) ** 2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, kernel_size=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(7, 7), padding=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid(),
        )

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.gru = ConvGRU(128, 128 + 128 + 64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """update operation"""

        batch, num, ch, ht, wd = net.shape
        device = net.device

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=device, dtype=net.dtype)

        out_dim = (batch, num, -1, ht, wd)

        net = net.view(batch * num, -1, ht, wd)
        inp = inp.view(batch * num, -1, ht, wd)
        corr = corr.view(batch * num, -1, ht, wd)
        flow = flow.view(batch * num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*out_dim)
        weight = self.weight(net).view(*out_dim)

        delta = delta.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
        weight = weight.permute(0, 1, 3, 4, 2)[..., :2].contiguous()

        net = net.view(*out_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(device))
            return net, delta, weight, eta, upmask
        else:
            return net, delta, weight
