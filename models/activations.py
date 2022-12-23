import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sigmoid(nn.Module):
    def __init__(self, hard=False):
        super(Sigmoid, self).__init__()
        self.hard = hard

    def forward(self, x):
        if self.hard:
            return F.relu6(
                x + 3.
            ) * 0.16667
        else:
            return F.sigmoid(x)

class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()
    
    def forward(self, x):
        cdf = 0.5 * (1 + torch.tanh(
            (math.sqrt(2 / math.pi) * (x + 0.044715) * torch.pow(x, 3))
        ))

        return x * cdf
