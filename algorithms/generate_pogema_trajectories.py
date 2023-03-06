import d3rlpy
import gc
import numpy as np
import sys
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import discounted_cumsum


def distance_to_goal(obs):
   # print(obs[1].shape)
    ax, ay = np.asarray(obs[1].shape) // 2
    # print(obs[2])
    # exit()
    if np.sum(obs[2]) == 1:
        x, y = np.where(obs[2] == 1)
        return ((ax - x) ** 2 + (ay - y) ** 2) ** 0.5
    return 1000


def episode_rewards_by_obs(episode_observation):
    min_dist = 1000
    new_rewards = []
    last_screen = np.zeros_like(episode_observation[0][0])
    for obs in episode_observation:
        dist = distance_to_goal(obs[-3:])
        reward = -0.0035
        if dist < min_dist:
            min_dist = dist
            reward = 0.01
        new_rewards.append(reward)
    new_rewards[-1] = 1
    # print(new_rewards)
    #  exit()
    return new_rewards


def load_pogema_trajectories(use_image=True):
    dataset = d3rlpy.dataset.MDPDataset.load("./data/test_mixed_dataset_10_0_40000.h5")[:1000]
   # additional = d3rlpy.dataset.MDPDataset.load("./data/mixed_dataset_10_0_27910.h5")
  # dataset.extend(additional)
  #  dataset, test_episodes = train_test_split(dataset, test_size=0.02, shuffle=True)
    traj, traj_len = [], []

    data_, episode_step = defaultdict(list), 0
    episode_data = dict()
    stack_len = 3
    traj = []
    traj_len = []
    full_data = {"observations": []}
    episode_data = dict()
    for i in tqdm(range(len(dataset))):
        # mask = np.where(dataset[i].actions == 0)
        mask = np.where(dataset[i].actions >= 0)
        # print("Zero indexes: ", mask)
        # print("Other: ", mask2)
        # print("-------------")
        # continue
        #  print()
        episode = dataset[i]
        episode_len = len(episode.rewards[mask])
        if episode_len < 20: continue
        new_rewards = np.asarray(episode_rewards_by_obs(episode.observations))
        for j in range(episode_len, 20, -3):
            i = j - 20
            if use_image:
                episode_data["observations"] = episode.observations[mask][i:i + 20].reshape(-1, stack_len * 3, 21, 21)
                full_data["observations"] += episode.observations[mask][i:i + 20].tolist()
            else:
                episode_data["observations"] = episode.observations[mask][i:i + 20].reshape(-1, stack_len * 3 * 21 * 21)
                full_data["observations"] += episode.observations[mask][i:i + 20].reshape(-1,
                                                                                          stack_len * 3 * 21 * 21).tolist()
            episode_data["actions"] = episode.actions[mask][i:i + 20].reshape(-1, 1)
            episode_data["rewards"] = new_rewards[mask][i:i + 20]
            reward_mask = np.where(episode_data["actions"].ravel() == 0)
            if len(reward_mask[0]) != 0:
                #  print(reward_mask)
                _mask = np.where(episode_data["actions"].ravel() != 0)
            #  print(_mask)
            #   print("-------------")
            episode_data["rewards"][reward_mask] = -0.05
            #   episode_data["rewards"][-1] = 1.5
            episode_data["returns"] = discounted_cumsum(episode_data["rewards"], 0.01)
            episode_len = len(episode.rewards[mask][i:i + 20])
            traj.append(episode_data)
            traj_len.append(episode_len)
            episode_data = dict()

    gc.collect()
    print("Trajectories statistics: ")
    print("Mean: ", np.mean(traj_len))
    print("Median: ", np.median(traj_len))
    print("Max: ", np.max(traj_len))
    print("Min: ", np.min(traj_len))
   # full_data["observations"] = np.asarray(full_data["observations"])
    print("Aboba")
    mean_val = np.mean(full_data["observations"][:5])
    std_val = np.mean(full_data["observations"][:5])
    obs_shape = traj[0]['observations'].shape[1]
    info = {
        "obs_mean": np.asarray([mean_val for _ in range(obs_shape)]).reshape(1, -1),
        "obs_std": np.asarray([std_val + 1e-6 for _ in range(obs_shape)]).reshape(1, -1),
        "traj_lens": np.array(traj_len),
    }
    print(sys.getsizeof(traj))
    print(traj[0]["observations"].nbytes * len(traj))
    return traj, info


if __name__ == "__main__":
    traj, info = load_pogema_trajectories(use_image=False)
    print(traj[0]['observations'].shape)
    print(traj[0]['actions'].shape)
    print(traj[0]['returns'].shape)
    print(traj[0]['rewards'].shape)

    print(info['obs_mean'].shape)
    print(info['traj_lens'].shape)
    print(traj[1]['actions'])
    print(traj[20]['actions'])
    from collections import Counter

    act = traj[0]['actions'].ravel().tolist()
    print(Counter(act))
    act = traj[1]['actions'].ravel().tolist()
    print(Counter(act))
    act = traj[2]['actions'].ravel().tolist()
    print(Counter(act))
    act = traj[3]['actions'].ravel().tolist()
    print(Counter(act))
    act = traj[53]['actions'].ravel().tolist()
    print(Counter(act))
