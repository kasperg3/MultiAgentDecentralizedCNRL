import argparse
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

# from https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DQN

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims[0], 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval'+'_lr'+str(self.lr),
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next'+'_lr'+str(self.lr),
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


import numpy as np
import matplotlib.pyplot as plt
import gym


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

import gym_mergablerobots

def main(args):

    env = gym.make('Concept-v0')
    best_score = -np.inf
    load_checkpoint = False

    # Get the adjustable parameters
    n_games = args.episodes
    save_freq = args.save_freq
    seed = args.seed
    learning_rate = args.lr
    load_checkpoint = args.load_model

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent0 = DQNAgent(gamma=0.98, epsilon=1.0, lr=learning_rate,
                     input_dims=(env.observation_space.shape[1],),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.02,
                     batch_size=64, replace=1000, eps_dec=0.0001,
                     chkpt_dir='models/', algo='DQNAgent0',
                     env_name='Concept-v0')

    agent1 = DQNAgent(gamma=0.98, epsilon=1.0, lr=learning_rate,
                     input_dims=(env.observation_space.shape[1],),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.02,
                     batch_size=64, replace=1000, eps_dec=0.0001,
                     chkpt_dir='models/', algo='DQNAgent1',
                     env_name='Concept-v0')

    agents = [agent0, agent1].copy()

    if load_checkpoint:
        for agent in agents:
            agent.load_models()

    fname = agent0.algo + '_' + agent0.env_name + '_lr' + str(agent0.lr) +'_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores_agent0, scores_agent1, eps_history, steps_array = [], [], [], []

    print("STARTING TRAINING WITH LR: " + str(learning_rate))

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = [0, 0]
        action = [5, 5]
        while not done:
            # Step in the environment
            observation_, reward, done, info = env.step(action)
            for agent in range(1):
                # if the action is done, add the transition to the replay buffer and learn
                agents[agent].store_transition(observation[agent], action[agent], reward[agent], observation_[agent], int(done))
                agents[agent].learn()
                # when the previous action is done, choose a new action
                action[agent] = agents[agent].choose_action(observation[agent])
                score[agent] += reward[agent]
            observation = observation_
            n_steps += 1

        # save the scores for each agent
        scores_agent0.append(score[0])
        scores_agent1.append(score[1])
        steps_array.append(n_steps)

        print(  'episode: ', i,
                ' | score: [%.2f %.2f]' % (score[0], score[1]),
                ' | epsilon agent[0,1]: [%.3f %.3f]' % (agents[0].epsilon, agents[1].epsilon),
                ' | steps', n_steps)

        if i % save_freq == 0:
            agents[0].save_models()
            agents[1].save_models()
            np.save(f"./results/agent0_scores_lr={str(learning_rate)}_history", scores_agent0)
            np.save(f"./results/agent1_scores_lr={str(learning_rate)}_history", scores_agent1)

        eps_history.append(agent0.epsilon)
        if load_checkpoint and n_steps >= 18000:
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--episodes", default=50000, type=int)  # Max time steps to run environment
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--save_freq", default=50, type=int)
    parser.add_argument("--load_model", action="store_true")  # Render the Training
    args = parser.parse_args()
    main(args)
