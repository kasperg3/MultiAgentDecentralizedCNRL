import argparse
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import gym_mergablerobots
import time

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

    def choose_action(self, observation, epsilon_random=True):

        if epsilon_random:
            # Choose a random action with epsilon greedy chance
            if np.random.random() > self.epsilon:
                state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
                actions = self.q_eval.forward(state)
                action = T.argmax(actions).item()
            else:
                action = np.random.choice(self.action_space)
        else:
            # No random action, choose the policy action
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
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


def evaluation(env, agents, episodes, dual_agent, agent_id=0):
    scores_agent0, scores_agent1, eps_history, steps_array, success_agent0, success_agent1 = [], [], [], [], [], []
    for episode in range(episodes):
        done = False
        observation = env.reset()
        score = [0, 0]
        success = [0, 0]
        action = [5, 5]
        while not done:
            # Step in the environment
            observation_, reward, done, info = env.step(action)
            if dual_agent:
                for agent in range(2):
                    action[agent] = agents[agent].choose_action(observation[agent], False)
                    score[agent] += reward[agent]
                    success[agent] = int(env.task_success(str(agent)))
            else:
                action[agent_id] = agents[agent_id].choose_action(observation[agent_id], False)
                score[agent_id] += reward[agent_id]
                success[agent_id] = int(env.task_success(str(agent_id)))
            observation = observation_

        # save the scores for each agent
        scores_agent0.append(score[0])
        scores_agent1.append(score[1])
        success_agent0.append(success[0])
        success_agent1.append(success[1])

    agent0scores = [np.mean(scores_agent0), np.mean(success_agent0)]
    agent1scores = [np.mean(scores_agent1), np.mean(success_agent1)]
    return agent0scores, agent1scores


def main(args):
    seconds_start = time.time()
    env = gym.make('Concept-v0')
    best_score = -np.inf
    load_checkpoint = False
    agent0returns_history = []
    agent0success_history = []
    agent1returns_history = []
    agent1success_history = []
    time_elapsed_history = []
    # Get the adjustable parameters
    n_games = args.episodes
    save_freq = args.save_freq
    eval_freq = args.eval_freq
    seed = args.seed
    learning_rate = args.lr
    note = args.note
    load_checkpoint = args.load_model
    eval_model = args.eval_model

    dual_agent = args.dual_robot
    agent_id = args.agent_id
    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent0 = DQNAgent(gamma=0.98, epsilon=1.0, lr=learning_rate,
                     input_dims=(env.observation_space.shape[1],),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.02,
                     batch_size=64, replace=1000, eps_dec=0.0001,
                     chkpt_dir='models/', algo=note+'DQNAgent0',
                     env_name='Concept-v0')

    agent1 = DQNAgent(gamma=0.98, epsilon=1.0, lr=learning_rate,
                     input_dims=(env.observation_space.shape[1],),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.02,
                     batch_size=64, replace=1000, eps_dec=0.0001,
                     chkpt_dir='models/', algo=note+'DQNAgent1',
                     env_name='Concept-v0')

    agents = [agent0, agent1].copy()

    if load_checkpoint or eval_model:
        for agent in agents:
            agent.load_models()

    fname = agent0.algo + '_' + agent0.env_name + '_lr' + str(agent0.lr) +'_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores_agent0, scores_agent1, eps_history, steps_array = [], [], [], []

    print("STARTING TRAINING WITH LR: " + str(learning_rate))
    print("note: " + note)

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = [0, 0]
        action = [5, 5]
        while not done:
            # Step in the environment
            observation_, reward, done, info = env.step(action)
            if dual_agent:
                for agent in range(len(agents)):
                    # if the action is done, add the transition to the replay buffer and learn
                    agents[agent].store_transition(observation[agent], action[agent], reward[agent], observation_[agent], int(done))
                    agents[agent].learn()
                    # when the previous action is done, choose a new action
                    action[agent] = agents[agent].choose_action(observation[agent], not eval_model)
                    score[agent] += reward[agent]
            else:
                agents[agent_id].store_transition(observation[agent_id], action[agent_id], reward[agent_id], observation_[agent_id], int(done))
                agents[agent_id].learn()
                action[agent_id] = agents[agent_id].choose_action(observation[agent_id], not eval_model)
                score[agent_id] += reward[agent_id]

            observation = observation_
            n_steps += 1

        # save the scores for each agent
        scores_agent0.append(score[0])
        scores_agent1.append(score[1])
        steps_array.append(n_steps)

        print(  'episode: ', i,
                ' | score: [%.2f %.2f]' % (score[0], score[1]),
                ' | epsilon agent: [%.3f %.3f]' % (agents[0].epsilon, agents[1].epsilon),
                ' | steps', n_steps)

        if i % save_freq == 0:
            # Only save the agent/agents which are being trained
            if dual_agent:
                agents[0].save_models()
                agents[1].save_models()
                np.save(f"./results/{note}_agent0_scores_lr={str(learning_rate)}_history", scores_agent0)
                np.save(f"./results/{note}_agent1_scores_lr={str(learning_rate)}_history", scores_agent1)
            else:
                agents[agent_id].save_models()
                np.save(f"./results/{note}_agent{agent_id}_scores_lr={str(learning_rate)}_history", scores_agent0)

        if (i+1) % eval_freq == 0:
            # Save scores
            agent0score, agent1score = evaluation(env, agents, 40, dual_agent, agent_id)  # scores given as (mean_episode_score, mean_success_rate)
            agent0returns_history.append(agent0score[0])
            agent0success_history.append(agent0score[1])
            agent1returns_history.append(agent1score[0])
            agent1success_history.append(agent1score[1])

            # Log time
            seconds_end = time.time()
            time_elapsed_history.append((seconds_end - seconds_start)/60)

            # Save models

            if dual_agent:
                np.save(f"./results/{note}_agent0_test_return_lr={str(learning_rate)}", agent0returns_history)
                np.save(f"./results/{note}_agent1_test_return_lr={str(learning_rate)}", agent1returns_history)
                np.save(f"./results/{note}_agent0_test_success_lr={str(learning_rate)}", agent0success_history)
                np.save(f"./results/{note}_agent1_test_success_lr={str(learning_rate)}", agent1success_history)
                np.save(f"./results/{note}_time_elapsed_history_lr={str(learning_rate)}", time_elapsed_history)
            elif agent_id == 1:
                np.save(f"./results/{note}_agent1_test_return_lr={str(learning_rate)}", agent1returns_history)
                np.save(f"./results/{note}_agent1_test_success_lr={str(learning_rate)}", agent1success_history)
                np.save(f"./results/{note}_time_elapsed_history_lr={str(learning_rate)}", time_elapsed_history)
            elif agent_id == 0:
                np.save(f"./results/{note}_agent0_test_return_lr={str(learning_rate)}", agent0returns_history)
                np.save(f"./results/{note}_agent0_test_success_lr={str(learning_rate)}", agent0success_history)
                np.save(f"./results/{note}_time_elapsed_history_lr={str(learning_rate)}", time_elapsed_history)

            print("AGENT0 | eval_return: ", agent0score[0], " | eval_success: ", agent0score[1], " | minutes trained: ", (seconds_end - seconds_start)/60)
            print("AGENT1 | eval_return: ", agent1score[0], " | eval_success: ", agent1score[1], " | minutes trained: ", (seconds_end - seconds_start)/60)
        eps_history.append(agent0.epsilon)
        if load_checkpoint and n_steps >= 18000:
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="sparse", type=str)
    parser.add_argument("--seed", default=1000, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--episodes", default=1500, type=int)  # Max time steps to run environment
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--save_freq", default=50, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--agent_id", default=0, type=int)
    parser.add_argument("--load_model", action="store_true")  # Load a existing model
    parser.add_argument("--eval_model", action="store_true")  #Evaluate a existing model
    parser.add_argument("--dual_robot", action="store_true")
    #parser.set_defaults(eval_model=False)       # change this to true for model evaluation

    args = parser.parse_args()
    main(args)
