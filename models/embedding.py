import torch.nn as nn

from models.activations import GeLU

__all__ = ['CNN']

class ConvNet(nn.Module):
    '''Convolutional Network Model builder'''
    def __init__(self, H, W):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            GeLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # calcualte new H and new W
        H = H // 2
        W = W // 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            GeLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        H = H // 2
        W = W // 2

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
