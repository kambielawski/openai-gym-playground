import gym
from gym import spaces
import pygame
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps":4}

    def __init__(self, size=5):
        # size of the square grid
        self.size = size
        # window size during render
        self.window_size = 512

        # both the agent and the target have a 2D position on the grid
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size-1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size-1, shape=(2,), dtype=int)
            }
        )

        # action space is discrete of size 4
        # meaning the agent has exactly 4 discrete actions it can take
        self.action_space = spaces.Discrete(4)

        # mapping from action space to 2D movement encoding
        self._action_to_direction = {
            0: np.array([1,0]),
            1: np.array([0,1]),
            2: np.array([-1,0]),
            3: np.array([0,-1])
        }

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location,ord=1)
        }

    def reset(self, seed=None, return_info=False, options=None):
        # reset using superclass (Gym's builtin reset)
        super().reset(seed=seed)

        # randomize where agent begins
        self._agent_location = self.np_random.integers(0, self.size, size=2)

        # randomize target location, ensuring agent and target don't collide
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2)

        # return environment observation and info
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation


    def step(self, action):
        direction = self._action_to_direction[action]
        # np.clip ensures agent stays within grid
        self._agent_location = np.clip(
            self._agent_location + direction, 
            a_min=0, 
            a_max=self.size-1
        )
        # detect if the simulation is considered done at each time step
        done = np.array_equal(self._agent_location, self._target_location)
        # generate a reward at each time step 
        reward = 1 if done else 0
        # get observation and info from environment
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, done, info

    '''
    We have to render the environment somehow.
    Here we use Pygame, but any other rendering library could
    be used (e.g. Pybullet!) 
    '''
    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255,255,255))
        # size of single grid square
        pix_square_size = (self.window_size / self.size)

        # draw target on canvas (red square)
        pygame.draw.rect(
            canvas, 
            (255,0,0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # draw agent on canvas (blue circle)
        pygame.draw.circle(
            canvas, 
            (0,0,255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # add gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0,pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1,0,2),
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


