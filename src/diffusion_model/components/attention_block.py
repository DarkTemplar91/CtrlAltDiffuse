from torch import nn
import torch


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q = self.q(x).reshape(batch, channels, -1)
        k = self.k(x).reshape(batch, channels, -1)
        v = self.v(x).reshape(batch, channels, -1)

        attention = self.softmax(torch.bmm(q.permute(0, 2, 1), k))
        attended_features = torch.bmm(v, attention.permute(0, 2, 1))
        attended_features = attended_features.reshape(batch, channels, height, width)
        return attended_features + x
