from os import name
import torch
import torch.nn as nn

UNITS = 128
MAX_STEPS = 50

class Critic(nn.Module):
    '''Actor Dynamic randomization'''
    def __init__(self, dim_state, dim_goal, dim_action, dim_env, env, tau, learning_rate, batch_size):
        super(Critic, self).__init__()
        # 超参数
        self._dim_state = dim_state
        self._dim_action = dim_action
        self._dim_goal = dim_goal
        self._action_bound = env.action_space.high
        self._internal_memory = []
        self._tau = tau
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._dim_env = dim_env

        # 设定网络层数结构
        self.ff_branch_l = nn.Linear(self._dim_env + self._dim_goal + self._dim_action + self._dim_state, UNITS)
        self.recurrent_branch_l = nn.LSTM(self._dim_state + self._dim_action, UNITS, num_layers=1, batch_first=True)  # batch_first代表第一个维度是 ？也就是不定批次数量 （29， 128，1层，True）
        self.merged_l1 = nn.Linear(UNITS * 2 , UNITS)
        self.merged_l2 = nn.Linear(UNITS, UNITS)
        self.out_l = nn.Linear(UNITS, 1)
    
    def forward(self, input_state, input_goal, input_action, input_env_parms, input_memory):
        '''建立网络结构
        input_state       (?, 25)
        input_goal        (?, 3)
        input_action      (?, 4)
        input_env         (?, 1)
        input_memory(?, 50, 29)  (N, L, H_in)
        '''
        input_ff = torch.cat([input_env_parms, input_goal, input_action, input_state], dim=1)     # (?, 33)
        ff_branch = torch.relu(self.ff_branch_l(input_ff))                                        # (?, 128)
        recurent_branch,_ = self.recurrent_branch_l(input_memory)                                 # (?, 50, 128)
        #取最后一时刻的数据即可
        recurent_branch = recurent_branch[:,-1,:]                                                 # (?, 128)
        merged_branch_input = torch.cat([ff_branch, recurent_branch], dim=1)                      # (?, 256)
        merged_branch = torch.relu(self.merged_l1(merged_branch_input))                           # (?, 128)
        merged_branch = torch.relu(self.merged_l2(merged_branch))                                 # (? ,128)
        out = self.out_l(merged_branch)                                                           # (?, 1)
        return out

# if __name__=='__main__':
#     import os
#     import numpy as np
#     import random
#     import gym
#     from environment import RandomizedEnvironment
#     from replay_buffer import Episode, ReplayBuffer

#     randomized_environment = RandomizedEnvironment('FetchSlide2-v1', [0.0, 1.0], [])
#     # generate an environment
#     randomized_environment.sample_env()
#     env, env_params = randomized_environment.get_env()
#     # reset the environment
#     current_obs_dict = env.reset()
#     # read the current goal, and initialize the episode
#     goal = current_obs_dict['desired_goal']
#     episode = Episode(goal, env_params, MAX_STEPS)
#     # get the first observation and first fake "old-action"
#     # TODO: decide if this fake action should be zero or random
#     obs = current_obs_dict['observation']
#     achieved = current_obs_dict['achieved_goal']
#     last_action = env.action_space.sample()
#     reward = env.compute_reward(achieved, goal, 0)
#     episode.add_step(last_action, obs, reward, achieved)
#     done = False

#     critic = Critic(25, 3, 4, 1, env, 5e-3, 1e-3, 1600)

#     obs = current_obs_dict['observation']
#     history = episode.get_history()
#     obs = obs.reshape(1, 25)
#     goal = goal.reshape(1, 3)
#     history = history.reshape(1, history.shape[0], history.shape[1])   # (1, 50, 29)

#     obs = torch.FloatTensor(obs)
#     goal = torch.FloatTensor(goal)
#     history = torch.FloatTensor(history)
#     action = torch.FloatTensor(env.action_space.sample().reshape(1, 4))
#     env_friction = torch.FloatTensor(env_params.reshape(1, 1))

#     # 尝试创建graph查看情况
#     from torch.utils.tensorboard import SummaryWriter
#     writer = SummaryWriter('critic_construction')
#     writer.add_graph(critic, [obs, goal, action, env_friction, history])

#     out = critic(obs, goal, action, env_friction, history)
        









        




