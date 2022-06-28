import gym
from gym.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="envs:GridWorldEnv",
    max_episode_steps=200,
)
