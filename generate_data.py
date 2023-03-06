import numpy as np

import d3rlpy
from d3rlpy.algos import DiscreteCQL
from d3rlpy.algos import DiscreteBCQ
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
import os
import sys
import pickle

sys.path.append("pogema-appo")
from pomapf.wrappers import MatrixObservationWrapper
import gym
# noinspection PyUnresolvedReferences
import pomapf
from pogema import GridConfig
from tqdm import tqdm
from generate_offline_pogema import example, load_maps
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str)
    parser.add_argument('framestack_size', type=int)
    args = parser.parse_args()
    
    grid_kind = load_maps() 
    os.makedirs("./observations_", exist_ok=True) 
    os.makedirs(f"./observations_/{args.file_name}", exist_ok=True) 
    full_obs, full_actions, full_reward, full_trajectories = [], [], [], [] 
    for i in tqdm(range(2000)):
        grid = grid_kind[i%3]
#         obs, actions, reward, trajectories = example(path_to_log = args.file_name, 
#                                                          grid = grid, dat_name = i,\
#                                                          obs_shape = (1,3* 21* 21), \
#                                                          num_agents = 3, \
#                                                          total_steps_needed = 1000,\
#                                                          framestack_size= args.framestack_size)

        with open(f'observations_/{args.file_name}//{i}.pickle', 'rb') as f:
            obs, actions, reward, trajectories = pickle.load(f)
        
        full_obs.append(obs)
        full_actions.append(actions)
        full_reward.append(reward)
        full_trajectories.append(trajectories)
        
    full_obs =  np.concatenate(full_obs, axis=0)
    full_reward =  np.concatenate(full_reward, axis=0)
    full_trajectories = np.concatenate(full_trajectories, axis=0)
    full_actions = np.concatenate(full_actions, axis = 0)
    
    with open(f'observations_/{args.file_name}//{args.file_name}.pickle', 'wb') as f:
        pickle.dump([full_obs, full_actions, full_reward, full_trajectories], f)
        

