import torch
import torch.nn as nn

__all__ = ['CNN']

class MLP(nn.Module):
    '''Convolutional Network Model builder'''
    def __init__(self, H, W, num_classes=10):
        super(MLP, self).__init__()
        self.fc = nn.Linear(H * W * 32, num_classes)

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = self.fc(out)
        return out
