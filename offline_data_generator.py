from agents.prioritzied import Prioritized, PrioritizedConfig
from agents.replan import RePlanConfig, RePlan
import gym
# noinspection PyUnresolvedReferences
import pomapf
from pogema import GridConfig
import h5py
from pomapf.wrappers import MatrixObservationWrapper
import numpy as np
import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
import h5py

def drop_global_information(observations):
    for agent_idx, obs in enumerate(observations):
        del obs['global_target_xy']
        del obs['global_xy']
        del obs['global_obstacles']
    return observations


def example(ind = 0, saver = 'h5py', data_count = 500, log_interval = 100,max_data = 10000, image = False, stack_len = 1):    
    stack_len = stack_len
    agents = 3
    radius = 10
    seed = None
    img_size = radius*2 + 1
    if image:
        obs_shape =( 3*stack_len,img_size,img_size,)
    else:
        obs_shape =( 3*stack_len*img_size*img_size,)
    gc = GridConfig(seed=seed, num_agents=agents, max_episode_steps=64, obs_radius=radius, size=16, density=0.3)
    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=False, auto_reset=False, egocentric_idx=None,
                   observation_type='MAPF')
    algo = Prioritized(PrioritizedConfig())
    start_point = 0
    
    agent_expirience = {'observations': [[]]*agents, 'next_observations': [[]]*agents,
                        'actions': [[]]*agents, 'rewards': [[]]*agents, 'terminals': [[]]*agents}
  #  max_data = 10000
    with h5py.File('data%d.hdf5'%ind, 'w') as f:
        observations_ = f.create_dataset('observations', (100,*obs_shape), maxshape=(max_data,*obs_shape))
        next_observations_ = f.create_dataset('next_observations', (100,*obs_shape), maxshape=(max_data,*obs_shape))
        actions_ = f.create_dataset('actions', (100,1), maxshape=(max_data,1))
        rewards_ = f.create_dataset('rewards', (100,), maxshape=(max_data,))
        terminals_ = f.create_dataset('terminals', (100,), maxshape=(max_data,))
        count_od_done = 0
        count_of_reward = 0
        for k in range(data_count):
            if k%log_interval == 0:
                print(f"{k}/{data_count}")
            observations = env.reset()
            algo.after_reset()
            dones = [False, ...]
            obs = MatrixObservationWrapper.to_matrix(observations)
            agent_expirience = {'observations': [[]]*agents, 'next_observations': [[]]*agents,
                        'actions': [[]]*agents, 'rewards': [[]]*agents, 'terminals': [[]]*agents}
            obs_ = obs[0]['obs'].reshape(-1, img_size, img_size)
            stack = [[np.zeros_like(obs_)]*(stack_len-1) for i in range(agents)]
            
            for j in range(agents):
               #     print(j)
                obs_ = obs[j]['obs']
                obs_ = np.asarray(obs_)
                if not image: obs_ = obs_.reshape(-1, )
                stack[j].append(obs_)
                
            while not all(dones):
                action = algo.act(observations, None, dones)
                action = np.asarray(action)
                action[np.where(action == None)] = 0
                observations, rewards, dones, info = env.step(action)
                
                count_of_reward += np.sum(rewards)
                count_od_done += int(np.sum(np.array(dones))==3)*3
                
                obs = MatrixObservationWrapper.to_matrix(observations)
               # print(rewards[dones])
               # print(rewards)
                for j in range(agents):
                    if len( agent_expirience['terminals'][j])>0 and agent_expirience['terminals'][j][-1] == 1: continue
                    obs_ = obs[j]['obs']
                    plt.imshow(obs_[0])
                    obs_ = np.asarray(obs_)
                    if not image: obs_ = obs_.reshape(-1, img_size, img_size)
                    stack[j].append(obs_)
                    #if len(stack[j])%(stack_len+1) == 0:
                    step_stack = np.asarray(stack[j].copy())
                    obs_ = step_stack[:stack_len].reshape(obs_shape) 
                    next_obs = step_stack[1:stack_len+1].reshape(obs_shape) 
                    agent_expirience['observations'][j].append(obs_)
                    agent_expirience['next_observations'][j].append(next_obs)
                    agent_expirience['actions'][j].append([action[j]])
                    agent_expirience['rewards'][j].append(rewards[j])
                    agent_expirience['terminals'][j].append(dones[j])
                    stack[j] = stack[j][1:stack_len+1]
               # print(dones)
                algo.after_step(dones)

            add_data = 0
            for j in range(agents):
                add_data += len(agent_expirience['rewards'][j])
          #  print()
            observations_.resize((start_point+add_data,*obs_shape))
            next_observations_.resize((start_point+add_data,*obs_shape))
            actions_.resize((start_point+add_data,1))
            rewards_.resize((start_point+add_data,))
            terminals_.resize((start_point+add_data,))
            
            end_point = start_point            
            for i in range(agents):
                    end_point += len(agent_expirience['observations'][i])
                    observations_[start_point:end_point] = agent_expirience['observations'][i]
                    next_observations_[start_point:end_point] = agent_expirience['next_observations'][i]
                    actions_[start_point:end_point] = agent_expirience['actions'][i]
                    rewards_[start_point:end_point] = agent_expirience['rewards'][i]
                    terminals_[start_point:end_point] = agent_expirience['terminals'][i]
                    start_point = end_point
     #   print(count_od_done,count_of_reward)
        return (next_observations_[:,:], actions_[:,:],
            rewards_[:], terminals_[:])
    return

if __name__ == '__main__':
    # p = Pool(2)
    # a = datetime.datetime.now()
  #   with p:
     STACK_LEN = 1
     observations,actions,rewards,terminals = example( data_count = 40000, log_interval = 1000, max_data = 100000000, image = True, stack_len = STACK_LEN)
    # b = datetime.datetime.now()
   #  print(b-a)


