import numpy as np
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


class ObsActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
      #  print(len(self.config.MOVES))
        #self.action_space = gym.spaces.Box(0.0, 1.0, shape = (1,))
        full_size = self.config.obs_radius * 2 + 1
        #obs for img 
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3,full_size, full_size,))
        #obs for line 
        #self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3*full_size*full_size,))

    def reset(self, **kwargs):
        obs = MatrixObservationWrapper.to_matrix(self.env.reset())
        agents = len(obs)
        obs_ = []
        for i in range(agents):
            ob = obs[i]['obs']
            ob = np.asarray(ob)
            ob = ob.astype(np.uint8)
            #obs_.append(ob.reshape(-1,))
            obs_.append(ob)
        return np.asarray(obs_)
    
    def get_normalized_score(self,score):
        return score

    def step(self, action):
      #  print("Actions for pogema: ", action)
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
        return np.asarray(obs_), reward[0], done[0], info[0]

class RewardLogger(gym.Wrapper):
    total_episodes = 0
    def __init__(self, env):
        super().__init__(env)
        self.total_reward = 0
        self.episodes = 0
        self.reward = 0
        
    def reset(self, **kwargs):
        self.total_episodes += 1
        self.episodes += 1
        
        if self.total_episodes%10 == 0:
            wandb.log({"env_wrapper_reward": self.total_reward/self.episodes,
                       "total_episodes":self.total_episodes})
            self.total_reward = 0
            self.episodes = 0
       # else:
        wandb.log({"reward":self.reward, "total_episodes":self.total_episodes})
        self.reward = 0
        return self.env.reset()
    
    def get_normalized_score(self,score):
        return score

    def step(self, action):
      #  print("Actions for pogema: ", action)

        observations, reward, done, info = self.env.step(action)
        self.total_reward += reward       
        self.reward += reward
        return  observations, reward, done, info
    
def reshape_obs(obs):
    #in - N agents 3 21 21
    # out - agents N*3 21 21
    
    chanels = obs.shape[-3]
    stack = obs.shape[0]
    size = obs.shape[-1]
  #  print(obs.shape)
    result  = []
    for i in range(obs.shape[1]):
        result.append(obs[:,i].reshape(chanels*stack, size, size))
       # print(obs[0,i,:,:,:].shape)
    return np.asarray(result)

class RavelFrameStack(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frame_stack = self.env.observation_space.shape[0]
        self.agents = self.env.observation_space.shape[1]
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=( 3*self.frame_stack,
                                                                 self.env.observation_space.shape[-1],
                                                                self.env.observation_space.shape[-1]))
    
    def reset(self, **kwargs):
        obs  = self.env.reset()
        #print(np.asarray(obs).shape)
        return reshape_obs(np.asarray(obs))
    
    def step(self, action):
      #  print(action)
        observations, reward, done, info = self.env.step(action)
       # print(observations.shape)
      #  print(self.observation_space.shape)
        observations = reshape_obs(np.asarray(observations))
       # print(observations.shape)
        return observations, reward, done, info
    