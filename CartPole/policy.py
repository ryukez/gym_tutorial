import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, nInput: int, nHidden: int, nOutput: int):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(nInput, nHidden)
        self.fc2 = nn.Linear(nHidden, nOutput)

    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        return F.softmax(self.fc2(l1), dim=1)
