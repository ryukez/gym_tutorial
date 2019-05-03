import gym
from typing import List, NamedTuple
import matplotlib.pyplot as plt
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import policy


# Tuple for calculating loss
class Step(NamedTuple):
    confidence: Variable
    reward: float


if __name__ == '__main__':
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--path', default='results/')
    args = parser.parse_args()

    env = gym.make(args.env)
    assert len(env.observation_space.shape) == 1

    ### Consts ###############################
    nState = env.observation_space.shape[0]
    nAction = env.action_space.n
    nEpisode = 1000
    gamma = 1.0
    dir = args.path
    rewardsPath = dir + 'rewards.txt'
    modelPath = dir + 'policyNet.pth'
    ##########################################

    policyNet = policy.PolicyNet(nState, 20, nAction)

    # Load trained model
    if args.load:
        param = torch.load(modelPath)
        policyNet.load_state_dict(param)

    optimizer = optim.Adam(policyNet.parameters())
    episodeRewards = []

    for episode in range(nEpisode):
        state = env.reset()
        steps: List[Step] = []

        # Exploration loop
        done = False
        while not done:
            env.render()

            # Make Decision
            x = Variable(torch.Tensor([state]))
            probs = policyNet(x)

            action = np.random.choice(range(nAction), p=probs.detach().numpy()[0])
            nextState, reward, done, _ = env.step(action)

            steps.append(Step(probs[0][action], reward))
            state = nextState

        # Training
        loss = Variable(torch.tensor(1, dtype=torch.float32))
        for i, step in enumerate(steps):
            reward = sum([(gamma ** j) * s.reward for j, s in enumerate(steps[i:])])
            loss += -torch.log(step.confidence) * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Summary
        reward = sum([(gamma ** i) * step.reward for i, step in enumerate(steps)])
        episodeRewards.append(reward)

        print("==========================================")
        print("Episode: ", episode)
        print("Reward: ", reward)

    # Plot & Save
    plt.plot(range(nEpisode), episodeRewards)
    plt.show()

    with open(rewardsPath, 'w') as f:
        f.write(str(episodeRewards))

    torch.save(policyNet.state_dict(), modelPath)
    env.close()
