import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

from collections import deque, namedtuple

env = gym.envs.make("Breakout-v0")

"""
0 = no op 
1 = fire
2 = left
3 = right
"""
VALID_ACTIONS = [0, 1, 2, 3]