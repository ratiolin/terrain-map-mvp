import gymnasium as gym
import numpy as np


class Env:
    def __init__(self, noise=0.0):
        self.env = gym.make("CartPole-v1")
        self.noise = noise

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if self.noise > 0:
            obs = obs + np.random.normal(0, self.noise, size=obs.shape)
        return obs, reward, done
