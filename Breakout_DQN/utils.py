from typing import NamedTuple, Tuple

import numpy as np
import torch
import torchvision.transforms as T


class Step(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    nextState: torch.Tensor

class Batch(NamedTuple):
    state: torch.Tensor
    action: torch.LongTensor
    reward: torch.Tensor
    nextState: torch.Tensor


class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.stateTensor = torch.zeros((capacity, 4, 84, 84))
        self.actionTensor = torch.zeros(capacity).long()
        self.rewardTensor = torch.zeros(capacity)
        self.nextStateTensor = torch.zeros((capacity, 4, 84, 84))
        self.index = 0
        self.size = 0

    def push(self, step: Step) -> None:
        self.stateTensor[self.index] = step.state
        self.actionTensor[self.index] = torch.tensor(step.action)
        self.rewardTensor[self.index] = torch.tensor(step.reward)
        self.nextStateTensor[self.index] = step.nextState

        self.index = (self.index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, size: int) -> Batch:
        perm = torch.randperm(self.size)
        index = perm[:size]
        return Batch(
            self.stateTensor[index],
            self.actionTensor[index],
            self.rewardTensor[index],
            self.nextStateTensor[index]
        )


def preprocess(x: np.ndarray) -> np.ndarray:
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(84),
        T.Grayscale(),
        T.ToTensor()
    ])

    return transform(x[50:, :, :]).unsqueeze(0)
