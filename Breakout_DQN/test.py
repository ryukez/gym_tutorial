import gym
import torch
import time

import utils
import models
import agents

### Consts ###############################
N_EPISODE = 100
FRAME_SKIP = 2
##########################################

if __name__ == '__main__':
    env = gym.make('Breakout-v0')

    nAction = env.action_space.n

    Q = models.QNet(nAction)
    Q.load_state_dict(torch.load('results/model.pth', map_location='cpu'))
    Q.eval()

    agent = agents.Agent(nAction, Q)

    t = 0
    action = env.action_space.sample()
    for episode in range(N_EPISODE):
        print("episode: %d\n" % (episode + 1))

        observation = env.reset()
        state = torch.cat([utils.preprocess(observation)] * 4, 1)
        sum_reward = 0

        # Exploration loop
        done = False
        while not done:
            env.render()

            if t % FRAME_SKIP == 0:
                action = agent.getAction(state)

            observation, reward, done, _ = env.step(action)
            nextState = torch.cat([state.narrow(1, 1, 3), utils.preprocess(observation)], 1)

            state = nextState
            sum_reward += reward
            t += 1

            time.sleep(0.03)

        print("  reward %f\n" % sum_reward)

    env.close()
