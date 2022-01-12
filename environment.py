import gym
import numpy as np
import random

import fetch_slide_2

# hardcoded for fetch_slide_2, with only friction
class RandomizedEnvironment:
    """ Randomized environment class """
    def __init__(self, experiment, parameter_ranges, goal_range):
        self._experiment = experiment     # gym的名字
        self._parameter_ranges = parameter_ranges  # 参数随机化的范围
        self._goal_range = goal_range     
        self._params = [0]
        random.seed(123)

    def sample_env(self):
        mini = self._parameter_ranges[0]     # 随机化参数下界
        maxi = self._parameter_ranges[1]     # 上界
        pick = mini + (maxi - mini)*random.random()   # 随机化得到的参数

        self._params = np.array([pick])
        self._env = gym.make(self._experiment)
        self._env.env.reward_type="dense"
        self._env.set_property('object0', 'geom_friction', [pick, 0.005, .0001])   # 根据xml文件修改相应属性，修改的是mujocopy生成的py类model

    def get_env(self):
        """
            Returns a randomized environment and the vector of the parameter
            space that corresponds to this very instance
        """
        return self._env, self._params

    def close_env(self):
        self._env.close()

    def get_goal(self):
        return

