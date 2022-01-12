import gym
import torch
import torch.nn as nn
import numpy as np

from actor import Actor
from critic import Critic
from noise import OrnsteinUhlenbeckActionNoise

import os
import numpy as np
import random
import gym
from environment import RandomizedEnvironment
from replay_buffer import Episode, ReplayBuffer
import hopper_2

MAX_STEPS = 50
TAU = 5e-3
LEARNING_RATE = 1e-3

def update_net(model, target_model, tau=1.):
    '''更新目标网络'''
    for tar_param, param in zip(target_model.parameters(), model.parameters()):
        tar_param.data.copy_(param.data * tau + tar_param.data * (1.0 - tau))

class Agent:
    '''就是写一个pytorch的DDPG的结构即可'''
    def __init__(self, experiment, batch_size):
        self._dummy_env = gym.make(experiment)

        # High parms for this code 
        # 判断是什么环境，是字典形还是普通环境
        obs_judge = self._dummy_env.reset()
        if type(obs_judge) == dict:
            self._dim_state = self._dummy_env.observation_space['observation'].shape[0]
            self._dim_goal = self._dummy_env.observation_space['desired_goal'].shape[0]
        else:
            self._dim_state = self._dummy_env.observation_space.shape[0]

        self._dim_action = self._dummy_env.action_space.shape[0]
        self._dim_env = 1      # 随机化环境参数的维数，如果是只随机化摩擦力，那就是1维，随机化多个参数，就是多维。
        self._batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # agent noise
        self._action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self._dim_action))

        # 初始化网络
        self._actor = Actor(self._dim_state, self._dim_goal, self._dim_action, self._dummy_env, TAU, LEARNING_RATE, self._batch_size).to(self.device)
        self._critic = Critic(self._dim_state, self._dim_goal, self._dim_action, self._dim_env, self._dummy_env, TAU, LEARNING_RATE, self._batch_size).to(self.device)
        self._actor_target = Actor(self._dim_state, self._dim_goal, self._dim_action, self._dummy_env, TAU, LEARNING_RATE, self._batch_size).to(self.device)
        self._critic_target = Critic(self._dim_state, self._dim_goal, self._dim_action, self._dim_env, self._dummy_env, TAU, LEARNING_RATE, self._batch_size).to(self.device)

        # 初始化目标网络的权重
        update_net(self._actor, self._actor_target, tau=1.)
        update_net(self._critic, self._critic_target, tau=1.)

        # 优化器
        self.actor_opt = torch.optim.Adam(self._actor.parameters(),lr=self._actor._learning_rate)
        self.critic_opt = torch.optim.Adam(self._critic.parameters(),lr=self._critic._learning_rate)

        # 设置loss函数
        self.loss_function = torch.nn.MSELoss()

        # 初始化记录scalar的字典
        self.summaries = {}
    
    def get_action(self, obs, goal, history):
        '''根据actor 得到动作的函数'''
        obs = torch.FloatTensor(obs).to(self.device)
        goal = torch.FloatTensor(goal).to(self.device)
        history = torch.FloatTensor(history).to(self.device)
        action = self._actor(obs, goal, history)
        action = action.cpu().detach().numpy()
        return action
    
    def action_noise(self):
        return self._action_noise()
    
    def update_target_actor(self):
        update_net(self._actor, self._actor_target, TAU)
    
    def update_target_critic(self):
        update_net(self._critic, self._critic_target, TAU)

    def save_model(self, filename):
        torch.save(self._actor, filename)

    def load_model(self, filename):
        self._actor = torch.load(filename)

    def get_dim_state(self):
        return self._dim_state

    def get_dim_action(self):
        return self._dim_action
        
    def get_dim_env(self):
        return self._dim_env

    def get_dim_goal(self):
        return self._dim_goal


# if __name__ == '__main__':
#     dummy_env = gym.make('FetchSlide2-v1')
    
#     env2 = gym.make('HopperRandom-v1')

#     print(dummy_env.observation_space['observation'].shape[0])
#     print(dummy_env.observation_space['desired_goal'].shape[0])
#     print(env2.observation_space.shape[0])
    
        