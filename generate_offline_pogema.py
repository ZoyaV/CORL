from agents.prioritzied import Prioritized, PrioritizedConfig
from agents.replan import RePlanConfig, RePlan
import gym
# noinspection PyUnresolvedReferences
import pomapf
from pogema.animation import AnimationMonitor, AnimationConfig
from pogema import GridConfig, pogema_v0
import h5py
from pomapf.wrappers import MatrixObservationWrapper
import numpy as np
import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
import h5py
import yaml
from tqdm import tqdm
import pickle

def drop_global_information(observations):
    for agent_idx, obs in enumerate(observations):
        del obs['global_target_xy']
        del obs['global_xy']
        del obs['global_obstacles']
    return observations

def create_frame_stack(obs, window, new_shape = (3, 21, 21)):
    padding = window - 1
    obs = np.asarray(obs).reshape(-1,*new_shape)
    padding_obs_start = obs[0].copy().reshape(1, *new_shape)
    padding_obs_end = obs[-1].copy().reshape(1, *new_shape)
    
    # pading for start
    padded_obs = np.append(padding_obs_start, obs, axis = 0)
    for i in range(padding-1):
        padded_obs = np.append(padding_obs_start, padded_obs, axis = 0)
        
    # pading for end
    padded_obs = np.append(padded_obs, padding_obs_end, axis = 0)
    for i in range(padding-1):
        padded_obs = np.append(padded_obs, padding_obs_end, axis = 0)
        
    frame_stack = [padded_obs[i:i+window] for i in range(len(padded_obs)-window+1)]
    return frame_stack[:-padding]

def obs_to_framestack(observation, t, obs_shape = (3, 21, 21), framestack_size = 4):
    t = np.asarray(t)
    #print(t)
    truncates = list(np.where(t == True)[0]+1)
    truncates = [0]+truncates
    truncates = np.asarray(truncates)
    full_framestack = create_frame_stack(observation[truncates[0]: truncates[1]], window = framestack_size, new_shape = obs_shape )
    for i in range(1, len(truncates)-1):        
        framestask = create_frame_stack(observation[truncates[i]: truncates[i+1]], window = framestack_size, new_shape = obs_shape)
        full_framestack = np.append(full_framestack, framestask, axis = 0)
    return full_framestack
    
def generate_data(path_to_log, grid_config, dat_name, obs_shape = (3, 21, 21), num_agents = 3,
            total_steps_needed = 80000, log = False, framestack_size = 4, checkpoints = True):

    env = pogema_v0(grid_config=grid_config)
    print(env.observation_space)
    # set egocentric to None to show from all agents perspective
    env = AnimationMonitor(env=env, animation_config=AnimationConfig(egocentric_idx=0))
    algo = RePlan(RePlanConfig())
    
    obs_all_agent, framestack_obs_all_agent, reward_all_agent, done_all_agent, actions_all_agent = [], [], [], [], []
    for i in range(num_agents):
        obs_all_agent.append([])
        framestack_obs_all_agent.append([])
        reward_all_agent.append([])
        done_all_agent.append([])
        actions_all_agent.append([])
        
    # Collect data
    total_steps = 0
    pbar = tqdm(total = total_steps_needed)
    while total_steps <= total_steps_needed:
        obs = env.reset()
        observations = MatrixObservationWrapper.to_matrix(obs)
        algo.after_reset()
        dones = [False, ...]
        # One episode
        while not all(dones):
            action = algo.act(obs, None, dones)
            obs, rewards, dones, info = env.step(action)
            observations = MatrixObservationWrapper.to_matrix(obs)#['obs']
            algo.after_step(dones)            
            for i in range(num_agents):
                if len(done_all_agent[i])>1 and done_all_agent[i][-1]==1 and dones[i] == 1:
                    continue  
                total_steps += 1
                pbar.update(1)
                obs_all_agent[i].append(observations[i]['obs'])
                reward_all_agent[i].append(rewards[i])
                done_all_agent[i].append(dones[i])
                actions_all_agent[i].append(action[i])
    pbar.close()

    for i in range(num_agents):
        framestack_obs_all_agent[i] = obs_to_framestack(obs_all_agent[i], done_all_agent[i], 
                                                        obs_shape = obs_shape, framestack_size = framestack_size)        
    full_obs =  np.concatenate(framestack_obs_all_agent, axis=0)
    full_reward =  np.concatenate(reward_all_agent, axis=0)
    full_trajectories = np.concatenate(done_all_agent, axis=0)
    full_actions = np.concatenate(actions_all_agent, axis = 0)
    
    if log:
        print("Final observation shape: ", full_obs.shape)
        print("Final reward shape: ", full_reward.shape)
        print("Final trajectories shape:", full_trajectories.shape)
        print("Final actions shape: ", full_actions.shape)    
        print(f"Check reward/count of episode: {np.sum(full_reward)}/{ np.sum(full_trajectories)}")

    if checkpoints:
        with open(f'observations_/{path_to_log}/{dat_name}.pickle', 'wb') as f:
            pickle.dump([full_obs, full_actions, full_reward, full_trajectories], f)
    
    return full_obs, full_actions, full_reward, full_trajectories
                
            
            
            