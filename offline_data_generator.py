from agents.prioritzied import Prioritized, PrioritizedConfig
from agents.replan import RePlanConfig, RePlan
import gym
# noinspection PyUnresolvedReferences
import pomapf
from pogema import GridConfig

from pomapf.wrappers import MatrixObservationWrapper


def drop_global_information(observations):
    for agent_idx, obs in enumerate(observations):
        del obs['global_target_xy']
        del obs['global_xy']
        del obs['global_obstacles']


def example():
    gc = GridConfig(seed=None, num_agents=1, max_episode_steps=64, obs_radius=5, size=16, density=0.3)

    # turn off with_animation to speedup generation speed
    env = gym.make('POMAPF-v0', grid_config=gc, with_animations=True, auto_reset=False, egocentric_idx=None,
                   observation_type='MAPF')

    # fully centralized planning approach
    algo = Prioritized(PrioritizedConfig())
    # partially observable decentralized planning approach
    # algo = RePlan(RePlanConfig())

    for _ in range(10):
        observations = env.reset()
        algo.after_reset()
        dones = [False, ...]

        while not all(dones):
            action = algo.act(observations, None, dones)
            observations, rewards, dones, info = env.step(action)
            drop_global_information(observations)
            observations = MatrixObservationWrapper.to_matrix(observations)

            # you can take observation dones and reward information here
            # print(observations[0]['obs']), exit(0)
            algo.after_step(dones)


if __name__ == '__main__':
    example()
