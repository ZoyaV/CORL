import numpy as np

import d3rlpy
from d3rlpy.algos import DiscreteCQL
from d3rlpy.algos import DiscreteBCQ
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

import sys
# sys.path.append('utils/')
# from wandb_loggers import WandbLogger, prepare_logger
# from multiagent_d3rlpy import evaluate_on_environment_magent

sys.path.append("pogema-appo")
from pomapf.wrappers import MatrixObservationWrapper
import gym
# noinspection PyUnresolvedReferences
import pomapf
from pogema import GridConfig

from offline_data_generator import example

if __name__ == "__main__":
    STACK_LEN = 4
    observations,actions,rewards,terminals = example( data_count = 50000, log_interval = 1000, max_data = 1000000, image = True, stack_len = STACK_LEN)
