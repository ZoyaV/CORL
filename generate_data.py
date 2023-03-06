import d3rlpy
import numpy as np
import os
import pickle
import sys
from d3rlpy.algos import DiscreteBCQ, DiscreteCQL
from d3rlpy.metrics.scorer import average_value_estimation_scorer, discounted_sum_of_advantage_scorer, \
    evaluate_on_environment, td_error_scorer
from sklearn.model_selection import train_test_split

sys.path.append("pogema-appo")
from pomapf.wrappers import MatrixObservationWrapper
import gym
# noinspection PyUnresolvedReferences
import pomapf
from pogema import GridConfig
from tqdm import tqdm
from generate_offline_pogema import generate_data
import argparse
from pogema import GridConfig


def load_maps():
    with open('maps.yaml') as f:
        q = yaml.safe_load(f)
    return list(q.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str)
    parser.add_argument('framestack_size', type=int)
    parser.add_argument('load', type=bool, default=False)
    parser.add_argument('fixed_map', type=bool, default=False)
    parser.add_argument('one_map_steps', type=int, default=100)
    parser.add_argument('changes_count', type=int, default=100)
    args = parser.parse_args()

    os.makedirs("./observations_", exist_ok=True)
    os.makedirs(f"./observations_/{args.file_name}", exist_ok=True)
    full_obs, full_actions, full_reward, full_trajectories = [], [], [], []
    grid_kind = load_maps()

    for i in tqdm(range(args.changes_count)):
        if args.fixed_map:
            grid = grid_kind[i % 3]
            gc = GridConfig(seed=None, num_agents=num_agents, max_episode_steps=128, \
                            map=grid, obs_radius=10, auto_reset=False, observation_type='MAPF')
        else:
            gc = GridConfig(seed=None, num_agents=num_agents, max_episode_steps=128, \
                            obs_radius=10, auto_reset=False, observation_type='MAPF', size=16, density=0.3)
        if args.load:
            with open(f'observations_/{args.file_name}//{i}.pickle', 'rb') as f:
                obs, actions, reward, trajectories = pickle.load(f)
        else:
            obs, actions, reward, trajectories = generate_data(path_to_log=args.file_name,
                                                               grid_config=gc, dat_name=i, \
                                                               obs_shape=(1, 3 * 21 * 21), \
                                                               num_agents=3, \
                                                               total_steps_needed=args.one_map_steps, \
                                                               framestack_size=args.framestack_size)
        full_obs.append(obs)
        full_actions.append(actions)
        full_reward.append(reward)
        full_trajectories.append(trajectories)

    full_obs = np.concatenate(full_obs, axis=0)
    full_reward = np.concatenate(full_reward, axis=0)
    full_trajectories = np.concatenate(full_trajectories, axis=0)
    full_actions = np.concatenate(full_actions, axis=0)

    with open(f'observations_/{args.file_name}//{args.file_name}.pickle', 'wb') as f:
        pickle.dump([full_obs, full_actions, full_reward, full_trajectories], f)
