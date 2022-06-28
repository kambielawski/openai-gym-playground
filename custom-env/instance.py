import gym
import numpy as np
from envs.GridWorldEnv import GridWorldEnv

env = GridWorldEnv(size=8)
env.reset()

for _ in range(10):
    obs, reward, done, info = env.step(np.random.randint(0,4))
    env.render()
    if done:
        env.reset()

env.close()
