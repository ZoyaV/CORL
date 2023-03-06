import numpy as np
import h5py
import d3rlpy
import os

from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

import sys
sys.path.append('utils/')
#from wandb_loggers import WandbLogger, prepare_logger
#from multiagent_d3rlpy import evaluate_on_environment_magent

sys.path.append("pogema-appo")
from pomapf.wrappers import MatrixObservationWrapper
import gym
# noinspection PyUnresolvedReferences
import pomapf
from pogema import GridConfig
from gym.wrappers import FrameStack
import h5py
from d3rlpy.metrics.scorer import evaluate_on_environment
from argparse import ArgumentParser
from algorithms.pogema_wrappers import Obs1DActionWrapper, RavelFrameStack
#from model import CustomEncoderFactory
import wandb
import pickle

class RandActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        full_size = self.config.obs_radius * 2 + 1
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(12,full_size, full_size,))
    def reset(self, **kwargs):
        obs = self.env.reset()
        return np.asarray(obs[0])

    def step(self, action):
       # print(action)
        observations, reward, done, info = self.env.step([action])
        return observations[0], reward[0], done[0], info[0]

    
def prepare_original_mdp(file, count):
    dataset = h5py.File(file, "r")
   # count = count
    observations,actions,rewards,terminals = dataset['observations'][:count],dataset['actions'][:count],\
                                            dataset['rewards'][:count],dataset['terminals'][:count]
    dataset.close()
    observations = observations.astype(np.uint8)    
    return observations,actions,rewards,terminals

def prepare_original_mdp_pickle(file, count):
   # dataset = h5py.File(file, "r")
   # count = count
    with open('observations_/new_generator_fs4/new_generator_fs4.pickle', 'rb') as f:
        observations,actions,rewards,terminals = pickle.load(f)
    observations = observations.astype(np.uint8)    
    return observations,actions,rewards,terminals

def prepare_random(env, count):
    random_policy = d3rlpy.algos.DiscreteRandomPolicy()
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=count, env=env)
    random_policy.collect(env, buffer, n_steps=count)
    dataset = buffer.to_mdp_dataset()    
    return dataset.observations, dataset.actions, dataset.rewards, dataset.terminals


def mix_by_episode(original, random, prop = [0.9, 0.1], episode_count=1000):
    #print(prop)
    print("Count of original episode: ", len(np.where(original[-1] == 1)[0]))
    print("Count of random episode: ", 0)#len(np.where(random[-1] == 1)[0]))
    original_count = int(episode_count * prop[0])
    random_count =0# int(episode_count * prop[1])
    
    if prop[0] == 1:
       # last_index_orig = -1
        last_index_orig = np.where(original[-1] == 1)[0][original_count]
        data_to_use_orig = [d[:last_index_orig] for d in original]
    else:
    #print(random_count)
        last_index_orig = np.where(original[-1] == 1)[0][original_count]
        last_index_rand = np.where(random[-1] == 1)[0]
       # print(last_index_rand)
        last_index_rand = last_index_rand[random_count]
        #print(last_index_rand)
        data_to_use_orig = [d[:last_index_orig] for d in original]
        data_to_use_rand = [d[:last_index_rand] for d in random]

        print("Original terminals count: ", np.sum(data_to_use_orig[-1]))
        print("Random terminals count: ", np.sum(data_to_use_rand[-1]))
    
    dataset_orig = d3rlpy.dataset.MDPDataset(
    observations=data_to_use_orig[0],
    actions=data_to_use_orig[1],
    rewards=data_to_use_orig[2],
    terminals=data_to_use_orig[3],
            )
    if prop[-1] > 0:
        dataset_rand = d3rlpy.dataset.MDPDataset(
        observations=data_to_use_rand[0],
        actions= data_to_use_rand[1].ravel(),
        rewards=data_to_use_rand[2].ravel(),
        terminals=data_to_use_rand[3],
                )

        dataset_orig.extend(dataset_rand)
    print("Count of terminals: ", np.sum(dataset_orig.terminals))
    print("Reward sum: ", np.sum(dataset_orig.rewards))
    
    os.makedirs("./data", exist_ok=True) 
    dataset_orig.dump(f"./data/4d_dataset_{int(prop[0]*10)}_{int(prop[1]*10)}_{episode_count}.h5")
    return dataset_orig


if __name__ == "__main__":
    STACK_LEN = 1
    radius = 10
    img_size = radius*2 + 1
    gc = GridConfig(seed=None, num_agents=1, max_episode_steps=64, obs_radius=radius, size=16, density=0.3)
    original  = prepare_original_mdp_pickle("", count = 500000)
    mix_by_episode(original, [], prop = [1, 0.0], episode_count=55000)

