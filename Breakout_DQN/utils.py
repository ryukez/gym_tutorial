import random
from typing import List, NamedTuple

import numpy as np
import torch
import torchvision.transforms as T


class Step(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    nextState: torch.Tensor


class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: List[Step] = []
        self.index = 0
        
    def push(self, step: Step) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(step)
        else:
            self.memory[self.index] = step

        self.index = (self.index + 1) % self.capacity

    def sample(self, size: int) -> List[Step]:
        return random.sample(self.memory, size)


def preprocess(x: np.ndarray) -> np.ndarray:
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(84),
        T.Grayscale(),
        T.ToTensor()
    ])

    return transform(x[50:, :, :]).unsqueeze(0)
