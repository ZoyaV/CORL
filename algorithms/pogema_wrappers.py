import numpy as np

import gym
# noinspection PyUnresolvedReferences
import os
cwd = os.listdir()
print("--------------")
print(cwd)
print("--------------")
print("--------------")
print("--------------")
print("--------------")
import sys
sys.path.append(".")
sys.path.append("..")
import pomapf
from typing import Union
from pomapf.wrappers import MatrixObservationWrapper
from pogema import GridConfig, pogema_v0
from gym.wrappers import FrameStack
from generate_offline_pogema import load_maps
from pogema.animation import AnimationMonitor, AnimationConfig

class RewardLogger(gym.Wrapper):
    total_episodes = 0

    def __init__(self, env, logger):
        super().__init__(env)
        self.total_reward = 0
        self.episodes = 0
        self.reward = 0
        self.logger = logger

    def reset(self, **kwargs):
        self.total_episodes += 1
        self.episodes += 1
        if self.total_episodes % 10 == 0:
            self.logger.log({"env_wrapper_reward": self.total_reward / self.episodes,
                             "total_episodes": self.total_episodes})
            self.total_reward = 0
            self.episodes = 0
        # else:
        self.logger.log({"reward": self.reward, "total_episodes": self.total_episodes})
        self.reward = 0
        return self.env.reset()

    def get_normalized_score(self, score):
        return score

    def step(self, action):
        observations, reward, done, info = self.env.step(action)
        self.total_reward += reward
        self.reward += reward
        return observations, reward, done, info


class Obs1DActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        #TODO: 10 - is radius
        full_size = 10 * 2 + 1
        # obs for line
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3 * full_size * full_size,))

    def reset(self, **kwargs):
        obs = MatrixObservationWrapper.to_matrix(self.env.reset())
        agents = len(obs)
        obs_ = []
        for i in range(agents):
            ob = obs[i]['obs']
            ob = np.asarray(ob)
            ob = ob.astype(np.uint8)
            obs_.append(ob.reshape(-1, ))
        return np.asarray(obs_)

    def get_normalized_score(self, score):
        return score

    def step(self, action):
        #  print("Actions for pogema: ", action)
     #   if env.config.num_agents
        action = [action]
        action = np.asarray(action).astype(int)
        observations, reward, done, info = self.env.step(action)
        obs = MatrixObservationWrapper.to_matrix(observations)
        agents = len(obs)
        obs_ = []
        for i in range(agents):
            ob = obs[i]['obs']
            ob = np.asarray(ob)
            ob = ob.astype(np.uint8)
            obs_.append(ob.reshape(-1, ))
        return np.asarray(obs_), reward, done, info


class Obs2DActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        full_size = 10 * 2 + 1
        # obs for img
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3, full_size, full_size,))

    def reset(self, **kwargs):
        #  self.env.reset()
        obs = MatrixObservationWrapper.to_matrix(self.env.reset())

        agents = len(obs)
        obs_ = []
        for i in range(agents):
            ob = obs[i]['obs']
            ob = np.asarray(ob)
            ob = ob.astype(np.uint8)
            obs_.append(ob)
        return np.asarray(obs_)

    def get_normalized_score(self, score):
        return score

    def step(self, action):
     #   print("Actions for pogema: ", action)
        if not (isinstance(action, list) or  isinstance(action, np.ndarray)):
            action = [action]
        action = np.asarray(action).astype(int)
        observations, reward, done, info = self.env.step(action)
        obs = MatrixObservationWrapper.to_matrix(observations)
        agents = len(obs)
        obs_ = []
        for i in range(agents):
            ob = obs[i]['obs']
            ob = np.asarray(ob)
            ob = ob.astype(np.uint8)
            # obs_.append(ob.reshape(-1,))
            obs_.append(ob)
       # print(done)
        return np.asarray(obs_), reward, done, info


def reshape_obs(obs, image=True):
    # in - N agents 3 21 21
    # out - agents N*3 21 21
    chanels = obs.shape[-3]
    stack = obs.shape[0]
    size = obs.shape[-1]
    result = []
    for i in range(obs.shape[1]):
        if image:
            result.append(obs[:, i].reshape(chanels * stack, size, size))
        else:
            result.append(obs[:, i].reshape(chanels * stack * size * size))
    return np.asarray(result)


class RavelFrameStack(gym.Wrapper):
    def __init__(self, env, image_obs=True):
        super().__init__(env)
        self.frame_stack = self.env.observation_space.shape[0]
        self.agents = self.env.observation_space.shape[1]
        self.field_size = self.env.observation_space.shape[-1]
        # print(self.env.observation_space.shape[-1])
        self.image_obs = image_obs
        if image_obs:
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3 * self.frame_stack,
                                                                     self.field_size,
                                                                     self.field_size))
        else:
            self.observation_space = gym.spaces.Box(0.0, 1.0,
                                                    shape=(1, 3 * self.frame_stack * self.field_size * self.field_size))

    def reset(self, **kwargs):
        obs = self.env.reset()
        return reshape_obs(np.asarray(obs), image=self.image_obs)

    def step(self, action):
        observations, reward, done, info = self.env.step(action)
        observations = reshape_obs(np.asarray(observations), image=self.image_obs)
       # print(done)
        return observations, reward, done, info


class UseOneAgent(gym.Wrapper):
    def reset(self, **kwargs):
        return self.env.reset()[0]

    def step(self, action):
        observations, reward, done, info = self.env.step(action)
        return observations[0], reward, done, info


def init_imagebased_pogema(stack_len, radius, img_size, num_agents):
    # Init image pogema
    grid = load_maps()[0]
    gc = GridConfig(seed=None, num_agents=num_agents, max_episode_steps=128, \
                    map = grid, obs_radius=10, auto_reset=False, observation_type='MAPF') 

    env = pogema_v0(grid_config=gc)
    
    env = Obs2DActionWrapper(env)
    env = FrameStack(env, num_stack=stack_len)
    env = RavelFrameStack(env)
    return env


def init_vactorbased_pogema(stack_len, radius, img_size, num_agents, wandb = None, vis = False):
    # Init image pogema

#     gc = GridConfig(seed=None, num_agents=num_agents, max_episode_steps=64, obs_radius=radius, size=16, density=0.3)
#     env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None,
#                    observation_type='MAPF')
#     grid = load_maps()[0]
#     gc = GridConfig(seed=None, num_agents=num_agents, max_episode_steps=500, \
#                   map = grid, obs_radius=10, auto_reset=False, observation_type='MAPF') 
    
    gc = GridConfig(seed=None, num_agents=num_agents, max_episode_steps=128, \
                    obs_radius=10, auto_reset=False, observation_type='MAPF', size=16, density=0.3) 

    env = pogema_v0(grid_config=gc)
    if vis:
        env = AnimationMonitor(env=env, animation_config=AnimationConfig(egocentric_idx=None))
            
    env = Obs2DActionWrapper(env)
    env = FrameStack(env, num_stack=stack_len)

    env = RavelFrameStack(env, image_obs=False)
    if isinstance(wandb, type(None)):
        pass
       # env = RewardLogger(env, wandb)
    return env


if __name__ == "__main__":
    print("Test image wrappers!")
    stack_len = 4
    radius = 10
    img_size = radius * 2 + 1
    num_agents = 1

    # Init image pogema
    env = init_imagebased_pogema(stack_len, radius, img_size, num_agents)

    obs = env.reset()
    obs_ = np.asarray(obs)
    assert (obs_.shape == (num_agents, 12, 21, 21))
    print("2d <reset> observation is OK")

    obs, _, _, _ = env.step(1)
    obs_ = np.asarray(obs)
    assert (obs_.shape == (num_agents, 12, 21, 21))
    print("2d <step> observation is OK")
    print()
    # Init vector pogema
    print("Test vector wrappers!")
    env = init_vactorbased_pogema(stack_len, radius, img_size, num_agents)
    obs = env.reset()
    obs_ = np.asarray(obs)
    assert (obs_.shape == (num_agents, 12 * 21 * 21))

    print("1d <reset> observation is OK")
    obs, _, _, _ = env.step(1)
    obs_ = np.asarray(obs)
    assert (obs_.shape == (num_agents, 12 * 21 * 21))
    print("1d <step> observation is OK")
    print("Act space: ", env.action_space)
    env = UseOneAgent(env)
    obs = env.reset()
    obs_ = np.asarray(obs)
    print(obs_.shape)
    assert (obs_.shape == (12 * 21 * 21,))
    print("1d <use one agent> observation is OK")

    print("\n Test vector wrappers with 3 agent!")
    stack_len = 4
    radius = 10
    img_size = radius * 2 + 1
    num_agents = 3

    env = init_vactorbased_pogema(stack_len, radius, img_size, num_agents)
    obs = env.reset()
    obs_ = np.asarray(obs)
    assert (obs_.shape == (num_agents, 12 * 21 * 21))

    print("1d <reset> observation is OK")
    obs, _, _, _ = env.step([1,2,3])
    obs_ = np.asarray(obs)
    assert (obs_.shape == (num_agents, 12 * 21 * 21))
    print(obs_.shape)
    print("1d <step> observation is OK")
    print("Act space: ", env.action_space)
