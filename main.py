import os
import numpy as np
import random
import gym
import torch

from environment import RandomizedEnvironment
from agent import Agent
from replay_buffer import Episode, ReplayBuffer

EPISODES = 1000000

directory = "checkpoints"
experiment = "FetchSlide2-v1"
env = gym.make(experiment)

# Program hyperparameters
TESTING_INTERVAL = 200 # number of updates between two evaluation of the policy
TESTING_ROLLOUTS = 100 # number of rollouts performed to evaluate the current policy

# Algorithm hyperparameters
BATCH_SIZE = 32
BUFFER_SIZE = 1000
MAX_STEPS = 50 # WARNING: defined in multiple files...
GAMMA = 0.99
K = 0.8 # probability of replay with H.E.R.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the agent, both the actor/critic (and target counterparts) networks
agent = Agent(experiment, BATCH_SIZE*MAX_STEPS)

# Initialize the environment sampler
randomized_environment = RandomizedEnvironment(experiment, [0.0, 1.0], [])

# Initialize the replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE)

if not os.path.exists(directory):
    os.makedirs(directory)

for ep in range(EPISODES):
    print('this is ep ',ep)
    # generate a rollout

    # generate an environment
    randomized_environment.sample_env()
    env, env_params = randomized_environment.get_env()


    # reset the environment
    current_obs_dict = env.reset()

    # read the current goal, and initialize the episode
    goal = current_obs_dict['desired_goal']
    episode = Episode(goal, env_params, MAX_STEPS)

    # get the first observation and first fake "old-action"
    # TODO: decide if this fake action should be zero or random
    obs = current_obs_dict['observation']
    achieved = current_obs_dict['achieved_goal']
    last_action = env.action_space.sample()

    reward = env.compute_reward(achieved, goal, 0)

    episode.add_step(last_action, obs, reward, achieved)

    done = False

    # rollout the  whole episode
    while not done:
        obs = current_obs_dict['observation']
        history = episode.get_history()

        obs = obs.reshape(-1, agent._dim_state)
        goal = goal.reshape(-1, 3)
        history = history.reshape(-1, history.shape[0], history.shape[1])   # (?, 50, 29)
        noise = agent.action_noise()
        action = agent.get_action(obs, goal, history) + noise

        new_obs_dict, step_reward, done, info = env.step(action[0])

        new_obs = new_obs_dict['observation']
        achieved = new_obs_dict['achieved_goal']

        episode.add_step(action[0], new_obs, step_reward, achieved, terminal = done)

        current_obs_dict = new_obs_dict

    # store the episode in the replay buffer
    replay_buffer.add(episode)

    # replay the episode with HER with probability k
    if random.random() < K:
        new_goal = current_obs_dict['achieved_goal']
        replay_episode = Episode(new_goal, env_params, MAX_STEPS)
        for action, state, achieved_goal, done in zip(episode.get_actions(), episode.get_states(), episode.get_achieved_goals(), episode.get_terminal()):
            # compute the new reward
            step_reward = env.compute_reward(achieved_goal, new_goal, 0)

            # add the fake transition
            replay_episode.add_step(action, state, step_reward, achieved_goal, terminal = done)

        replay_buffer.add(replay_episode)

    # close the environment
    randomized_environment.close_env()

    # perform a batch update of the network if we can sample a big enough batch
    # from the replay buffer

    if replay_buffer.size() > BATCH_SIZE:
        episodes = replay_buffer.sample_batch(BATCH_SIZE)

        s_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_state()])
        a_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_action()])

        next_s_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_state()])

        r_batch = np.zeros([BATCH_SIZE*MAX_STEPS])

        env_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_env()])
        goal_batch = np.zeros([BATCH_SIZE*MAX_STEPS, agent.get_dim_goal()])

        history_batch = np.zeros([BATCH_SIZE*MAX_STEPS, MAX_STEPS, agent.get_dim_action()+agent.get_dim_state()])

        t_batch = []

        for i in range(BATCH_SIZE):
            s_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_states())[:-1]
            a_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_actions())[1:]
            next_s_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_states())[1:]
            r_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array(episodes[i].get_rewards())[1:]

            env_batch[i*MAX_STEPS:(i+1)*MAX_STEPS]=np.array(MAX_STEPS*[episodes[i].get_env()])
            goal_batch[i*MAX_STEPS:(i+1)*MAX_STEPS]=np.array(MAX_STEPS*[episodes[i].get_goal()])
            history_batch[i*MAX_STEPS:(i+1)*MAX_STEPS] = np.array([episodes[i].get_history(t = t) for t in range(1, MAX_STEPS+1)])

            # WARNING FIXME: needs padding
            t_batch += episodes[i].get_terminal()[1:]

        s_batch = torch.FloatTensor(s_batch).to(device)
        a_batch = torch.FloatTensor(a_batch).to(device)
        next_s_batch = torch.FloatTensor(next_s_batch).to(device)
        r_batch = torch.FloatTensor(r_batch).to(device)
        r_batch = r_batch.view(-1, 1)
        env_batch = torch.FloatTensor(env_batch).to(device)
        goal_batch = torch.FloatTensor(goal_batch).to(device)
        history_batch = torch.FloatTensor(history_batch).to(device)
        t_batch = torch.FloatTensor(t_batch).to(device)
        t_batch = t_batch.view(-1, 1)

        target_action_batch = agent._actor_target(next_s_batch, goal_batch, history_batch)
        target_q = agent._critic_target(next_s_batch, goal_batch, target_action_batch, env_batch, history_batch)

        y = (r_batch + (1.- t_batch) * GAMMA * target_q).detach()

        curr_q = agent._critic(s_batch, goal_batch, a_batch, env_batch, history_batch)

        loss_critic = agent.loss_function(curr_q, y)
        agent.critic_opt.zero_grad()
        loss_critic.backward()
        agent.critic_opt.step()

        # 计算actor的损失，并更新actor
        a_outs = agent._actor(s_batch, goal_batch, history_batch) 
        loss_actor = -agent._critic(s_batch, goal_batch, a_outs, env_batch, history_batch).mean()
        agent.actor_opt.zero_grad()
        loss_actor.backward()
        agent.actor_opt.step()

        # Update target networks
        agent.update_target_actor()
        agent.update_target_critic()


    # perform policy evaluation
    if ep % TESTING_INTERVAL == 0:
        success_number = 0
        
        for test_ep in range(TESTING_ROLLOUTS):
            randomized_environment.sample_env()
            env, env_params = randomized_environment.get_env()

            current_obs_dict = env.reset()

            # read the current goal, and initialize the episode
            goal = current_obs_dict['desired_goal']
            episode = Episode(goal, env_params, MAX_STEPS)

            # get the first observation and first fake "old-action"
            # TODO: decide if this fake action should be zero or random
            obs = current_obs_dict['observation']
            achieved = current_obs_dict['achieved_goal']
            last_action = env.action_space.sample()

            episode.add_step(last_action, obs, 0, achieved)

            done = False

            # rollout the whole episode
            while not done:
                obs = current_obs_dict['observation']
                history = episode.get_history()

                obs = obs.reshape(1, agent._dim_state)
                goal = goal.reshape(1, 3)
                history = history.reshape(1, history.shape[0], history.shape[1])  

                action = agent.get_action(obs,goal,history)

                new_obs_dict, step_reward, done, info = env.step(action[0])

                new_obs = new_obs_dict['observation']
                achieved = new_obs_dict['achieved_goal']

                episode.add_step(action[0], new_obs, step_reward, achieved, terminal=done)

                current_obs_dict = new_obs_dict

            if info['is_success'] > 0.0:
                success_number += 1

            randomized_environment.close_env()

        print("Testing at episode {}, success rate : {}".format(ep, success_number/TESTING_ROLLOUTS))
        agent.save_model("{}/ckpt_episode_{}.pth".format(directory, ep))

