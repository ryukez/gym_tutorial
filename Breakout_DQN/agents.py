import numpy as np
import torch
from torch.autograd import Variable

import models


class Agent(object):
    def __init__(self, nAction: int, Q: models.QNet) -> None:
        self.nAction = nAction
        self.Q = Q

    def getAction(self, state: torch.Tensor, eps: float) -> int:
        q, argq = self.Q(Variable(state).cuda()).max(1)

        probs = np.full(self.nAction, eps / self.nAction, np.float32)
        probs[argq[0]] += 1 - eps
        return np.random.choice(np.arange(self.nAction), p=probs)
