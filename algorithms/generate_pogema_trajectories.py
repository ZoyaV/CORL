sys.path.append("./pogema-appo")
print("--------------")
import os
cwd = os.getcwd()
print(cwd)
print("--------------")
from pomapf.wrappers import MatrixObservationWrapper
import gym
# noinspection PyUnresolvedReferences
import pomapf
from pogema import GridConfig
from gym.wrappers import FrameStack

from torch import nn
import torch 
import numpy as np

def load_pogema_trajectories(dataset):
    #dataset = gym.make(env_name).get_dataset()
    dataset = d3rlpy.dataset.MDPDataset.load("./algorithms/mixed_dataset_10_00_5000.h5")
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2, shuffle = True)
    traj, traj_len = [], []

    data_, episode_step = defaultdict(list), 0
    episode_data = dict()
    traj = []
    traj_len = []
    full_data = {"observations":[]}
    for i in range(len(dataset)):
        episode_data["observations"] = dataset[i].observations.reshape(-1,12, 21,21)
      #  raise Exception(episode_data["observations"].shape)
        full_data["observations"] +=  dataset[i].observations.tolist()
        episode_data["actions"] = dataset[i].actions
        episode_data["rewards"] = dataset[i].rewards
        episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=1
             )
        episode_len = len(dataset[i].rewards)

        traj.append(episode_data)
        traj_len.append(episode_len)
        
    full_data["observations"] =  np.asarray(full_data["observations"])
  #  print(full_data["observations"].shape)
    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": full_data["observations"].mean(0, keepdims=True),
        "obs_std": full_data["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info
