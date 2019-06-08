from typing import List

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

    def update(self, batch: List[utils.Step]) -> None:
        stateBatch = Variable(torch.cat([step.state for step in batch], 0).cuda())
        actionBatch = torch.LongTensor([step.action for step in batch]).cuda()
        rewardBatch = torch.Tensor([step.reward for step in batch]).cuda()
        nextStateBatch = Variable(torch.cat([step.nextState for step in batch], 0).cuda())

        qValue = self.Q(stateBatch).gather(1, actionBatch.unsqueeze(1)).squeeze(1)
        qTarget = rewardBatch + self.QTarget(nextStateBatch).detach().max(1)[0] * self.gamma
        
        L = self.lossFunc(qValue, qTarget)
        self.opt.zero_grad()
        L.backward()
        self.opt.step()
