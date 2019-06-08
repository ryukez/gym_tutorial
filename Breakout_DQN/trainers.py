import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import utils
import models

GAMMA = 0.99

class Trainer(object):
    def __init__(self, Q: models.QNet, QTarget: models.QNet, opt: optim.Optimizer) -> None:
        self.Q = Q
        self.QTarget = QTarget
        self.opt = opt

        self.gamma = GAMMA
        self.lossFunc = nn.SmoothL1Loss()

    def update(self, batch: utils.Batch) -> None:

        qValue = self.Q(Variable(batch.state)).gather(1, batch.action.unsqueeze(1)).squeeze(1)
        qTarget = batch.reward + self.QTarget(Variable(batch.nextState)).detach().max(1)[0] * self.gamma

        L = self.lossFunc(qValue, qTarget)
        self.opt.zero_grad()
        L.backward()
        self.opt.step()
