import gym
import numpy as np
from envs.GridWorldEnv import GridWorldEnv

env_funcs = [lambda: GridWorldEnv(size=i) for i in range(2,5)]

print(env_funcs)

envs = gym.vector.AsyncVectorEnv(env_funcs)

envs.reset()

observations, rewards, dones, infos = envs.step([np.random.randint(0,4) for _ in range(2,5)])
print(observations)
print(dones)

