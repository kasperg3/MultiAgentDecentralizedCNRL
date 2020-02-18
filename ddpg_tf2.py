import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

# Use the noise model implemented in openai baselines
import baselines.ddpg.noise as noise


# Classes:
#   Replay buffer
#   Noise model/models: They used Ornstein uhlenbeck
#   Actor
#   Critic
#   Agent Class as "glue" between DNN and env

# Uses batch norm and Adam optimizer

# We have a deterministic policy mu: input a state and outputs a probability of an action
# q learning uses random greedy to select random actions
# How should we handle the explore/exploit: Uses a stochastic policy to learn a greedy policy
# Deterministic means outputs the actual action instead of a probability,
# needs a way to bound the actions to the env limit

# uses a separate target network to calculate y_t

# dnn can be unstable, so we use a "target" network to train on and updates the real network.
# the actor critic network uses "soft" target updates
# the target networks are not just copied, but changed slowly by a factor tau:
# theta_prime = tau*theta + (1-tau)*theta_prime, tau << 1

# PARAMETERS USED IN DDPG ARTICLE:
#   L2 realization = 10^-2
#   10^-4 and 10^-3 to actor and critic respectively
#   gamma = 0.99    (Discount factor)
#   tau = 0.001     (target network update factor)
#   2 hidden layers, 400 and 300 units respectively
#   Noise: theta=0.15 and sigma=0.2

# The implementation of this agent is based of https://www.youtube.com/watch?v=jDll4JSI-xo
# with modification.


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_action):
        self.mem_size = max_size
        self.mem_pointer = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, *n_action))
        self.reward_memory = np.zeros(self.mem_size)

        # maybe add terminal memory, if the agent reaches a terminal state then the discount factor(gamma) == 0
        # Dont want to count the reward after it has ended, but is not necesarry in this application

    def save(self, state, action, reward, new_state):
        idx = self.mem_pointer % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = new_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward

    def save(self, dict):
        # TODO make it able to take dictionaries
        pass

    def sample_batch(self, batch_size):
        max_mem = min(self.mem_pointer, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        return states, new_states, actions, rewards


class Actor(object):
    def __init__(self, lr, n_actions, input_dimension, tf_session, action_boundary, name, dense1=400, dense2=300,
                 batch_size=64, checkpoint_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.input_dimension = input_dimension
        self.tf_session = tf_session
        self.action_boundary = action_boundary
        self.dense1 = dense1
        self.dense2 = dense2
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir

        # Build the TF network
        self.model = self.build_network()

        # Update the target network
        # TODO: Create a trainable variables scope, such that the target and actor is updated seperately
        self.params = tf.Variable(name=name)

        # Saver object to save model
        # TODO CREATE A SAVER

        # mu = action taken, parameters of mu, gradients of mu
        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.divide(x, self.batch_size),
                                        self.unnormalized_actor_gradients))
        self.optimize = tf.keras.optimizers.Adam(self.lr).apply_gradients(zip(self.actor_gradients, self.params))


    def build_network(self):
        f1 = 1/np.sqrt(self.dense1)
        f2 = 1/np.sqrt(self.dense2)
        f3 = 0.003

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(tf.float32, input_shape=[None, *self.input_dimension], name='inputs'),
            tf.keras.layers.Layer(tf.float32, input_shape=[None, *self.n_actions], name='gradients'),
            tf.keras.layers.Dense(units=self.dense1,
                                  bias_initializer=tf.keras.initializers.RandomUniform(minval=-f1, maxval=f1),
                                  kernel_initializer=tf.keras.initializers.RandomUniform(minval=-f1, maxval=f1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(units=self.dense2,
                                  bias_initializer=tf.keras.initializers.RandomUniform(minval=-f2, maxval=f2),
                                  kernel_initializer=tf.keras.initializers.RandomUniform(minval=-f2, maxval=f2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(units=self.n_actions,
                                  activation='tanh',
                                  bias_initializer=tf.keras.initializers.RandomUniform(minval=-f3, maxval=f3),
                                  kernel_initializer=tf.keras.initializers.RandomUniform(minval=-f3, maxval=f3)),
            tf.keras.layers.multiply(self.action_boundary)
        ])
        return model

    def predict(self, inputs):
        return self.model.predict(x=inputs)

    def train(self, inputs, gradients):
        tf.GradientTape
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


class Critic(object):
    pass


class Agent(object):
    pass
