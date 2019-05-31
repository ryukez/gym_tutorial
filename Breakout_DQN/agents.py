import numpy as np
import torch
from torch.autograd import Variable

import models

EPS = 0.05


class Agent(object):
    def __init__(self, nAction: int, Q: models.QNet) -> None:
        self.nAction = nAction
        self.Q = Q

        self.eps = EPS

    def getAction(self, state: torch.Tensor) -> int:
        q, argq = self.Q(Variable(state)).max(1)

        probs = np.full(self.nAction, self.eps / self.nAction, np.float32)
        probs[argq[0]] += 1 - self.eps
        return np.random.choice(np.arange(self.nAction), p=probs)
