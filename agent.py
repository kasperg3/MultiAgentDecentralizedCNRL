import gym
import numpy as np
import tensorflow as tf
import os
from replay import Memory


class Noise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=0.02):
        self.mu = mu
        self.theta = 0.15
        self.sigma = 0.2
        self.dt = dt
        self.reset()

    def get_noise(self):
        # compute noise using Euler-Maruyama method
        noise = self.noise_lag1 + self.theta * (self.mu - self.noise_lag1) * \
                self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        # set for next round
        self.noise_lag1 = noise
        return noise

    def reset(self):
        self.noise_lag1 = np.zeros_like(self.mu)

# From https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def get_noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)



class Actor(object):

    def __init__(self, session, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, hidden_size, chkpt_dir='tmp/'):
        self.sess = session
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.name = "actor"
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor' + '_ddpg.ckpt')

        # create actor
        self.inputs, self.outputs, self.scaled_outputs = self.create()
        self.actor_weights = tf.trainable_variables()

        # create target
        self.target_inputs, self.target_outputs, self.target_scaled_outputs = self.create()
        self.target_actor_weights = tf.trainable_variables()[len(self.actor_weights):]

        # set target weights to be actor weights using Polyak averaging
        self.update_target_weights = \
            [self.target_actor_weights[i].assign(tf.multiply(self.actor_weights[i], self.tau) +
                                                 tf.multiply(self.target_actor_weights[i], 1. - self.tau))
             for i in range(len(self.target_actor_weights))]

        # placeholder for gradient feed from critic -- i.e. critic comments
        self.comment_gradients = tf.placeholder(tf.float32, [None, self.action_dim])

        # combine actor gradients and comment gradients, then normalize
        self.unm_actor_gradients = tf.gradients(self.scaled_outputs, self.actor_weights, -self.comment_gradients)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unm_actor_gradients))

        # optimize using Adam
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.actor_weights))

        # count of weights
        self.n_actor_vars = len(self.actor_weights) + len(self.target_actor_weights)

    # function to create agent network
    def create(self):

        f1 = 1. / np.sqrt(self.hidden_size[0])
        f2 = 1. / np.sqrt(self.hidden_size[1])
        f3 = 1. / np.sqrt(self.hidden_size[2])
        f4 = 0.003
        inputs = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        x = tf.layers.dense(inputs, units=self.hidden_size[0],
                            kernel_initializer=tf.initializers.random_uniform(-f1, f1),
                            bias_initializer=tf.initializers.random_uniform(-f1, f1))
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, units=self.hidden_size[1],
                            kernel_initializer=tf.initializers.random_uniform(-f2, f2),
                            bias_initializer=tf.initializers.random_uniform(-f2, f2))
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)

        #Third
        x = tf.layers.dense(x, self.hidden_size[2],
                            kernel_initializer=tf.initializers.random_uniform(-f3, f3),
                            bias_initializer=tf.initializers.random_uniform(-f3, f3))
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)

        # activation layer
        outputs = tf.layers.dense(x, units=self.action_dim, activation='tanh',
                                  kernel_initializer=tf.initializers.random_uniform(-f4, f4),
                                  bias_initializer=tf.initializers.random_uniform(-f4, f4))

        # scale output fit action_bound
        scaled_outputs = tf.multiply(outputs, self.action_bound)
        return inputs, outputs, scaled_outputs

    # function to train by adding gradient and optimize
    def train(self, inputs, grad):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.comment_gradients: grad
        })

    # function to predict
    def predict(self, inputs):
        return self.sess.run(self.scaled_outputs, feed_dict={
            self.inputs: inputs
        })

    # function to predict target
    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_outputs, feed_dict={
            self.target_inputs: inputs
        })

    # function to update target
    def update(self):
        self.sess.run(self.update_target_weights)

# Critic Class
class Critic(object):

    def __init__(self, session, state_dim, action_dim, learning_rate, tau, gamma, n_actor_vars, hidden_size, chkpt_dir='tmp/'):
        self.sess = session
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic' + '_ddpg.ckpt')

        # create critic
        self.inputs, self.actions, self.outputs = self.create()
        self.critic_weights = tf.trainable_variables()[n_actor_vars:]

        # create target
        self.target_inputs, self.target_actions, self.target_outputs = self.create()
        self.target_critic_weights = tf.trainable_variables()[(len(self.critic_weights) + n_actor_vars):]

        # set target weights to be actor weights using Polyak averaging
        self.update_target_weights = \
            [self.target_critic_weights[i].assign(tf.multiply(self.critic_weights[i], self.tau) \
                                                  + tf.multiply(self.target_critic_weights[i], 1. - self.tau))
             for i in range(len(self.target_critic_weights))]

        # placeholder for predicted q
        self.pred_q = tf.placeholder(tf.float32, [None, 1])

        # optimize mse using Adam
        self.loss = tf.reduce_mean(tf.square(self.pred_q - self.outputs))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # comment gradients to feed actor
        self.comment_gradients = tf.gradients(self.outputs, self.actions)

    # function to create agent network
    def create(self):

        f1 = 1. / np.sqrt(self.hidden_size[0])
        f2 = 1. / np.sqrt(self.hidden_size[1])
        f3 = 1. / np.sqrt(self.hidden_size[2])
        f4 = 0.003

        # state branch
        inputs = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        x = tf.layers.dense(inputs, self.hidden_size[0],
                            kernel_initializer=tf.initializers.random_uniform(-f1, f1),
                            bias_initializer=tf.initializers.random_uniform(-f1, f1))
        x = tf.layers.batch_normalization(x)
        # x = tf.nn.relu(x)

        # action branch
        actions = tf.placeholder(tf.float32, shape=[None, self.action_dim])

        # merge
        x = tf.concat([x, actions], axis=1)
        x = tf.layers.dense(x, self.hidden_size[1],
                            kernel_initializer=tf.initializers.random_uniform(-f2, f2),
                            bias_initializer=tf.initializers.random_uniform(-f2, f2))
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)

        #Third
        x = tf.layers.dense(x, self.hidden_size[2],
                            kernel_initializer=tf.initializers.random_uniform(-f3, f3),
                            bias_initializer=tf.initializers.random_uniform(-f3, f3))
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)

        # activation layer
        outputs = tf.layers.dense(x, 1,
                                  kernel_initializer=tf.initializers.random_uniform(-f4, f4),
                                  bias_initializer=tf.initializers.random_uniform(-f4, f4))
        return inputs, actions, outputs

    # function to train by adding states, actions, and q values
    def train(self, inputs, actions, pred_q):
        return self.sess.run([self.outputs, self.optimize], feed_dict={
            self.inputs: inputs,
            self.actions: actions,
            self.pred_q: pred_q
        })

    # function to predict
    def predict(self, inputs, actions):
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: inputs,
            self.actions: actions
        })

    # function to predict target
    def predict_target(self, inputs, actions):
        return self.sess.run(self.target_outputs, feed_dict={
            self.target_inputs: inputs,
            self.target_actions: actions
        })

    # function to update target
    def update(self):
        self.sess.run(self.update_target_weights)

    # function to compute gradients to feed actor -- i.e. critic comment
    def get_comment_gradients(self, inputs, actions):
        return self.sess.run(self.comment_gradients, feed_dict={
            self.inputs: inputs,
            self.actions: actions
        })

class Agent(object):

    def __init__(self, args, sess, env):
        # Set path to save result
        gym_dir = './' + args['env'] + '_' + args['variation'] + '/gym'
        self.batch_size = int(args['batch_size'])

        # get size of action and state (i.e. output and input for the agent)
        obs = env.reset()
        observation_dim = obs['observation'].shape[0]
        achieved_goal_dim = obs['achieved_goal'].shape[0]
        desired_goal_dim = obs['desired_goal'].shape[0]
        assert achieved_goal_dim == desired_goal_dim

        # state size = observation size + goal size
        state_dim = observation_dim + desired_goal_dim
        action_dim = env.action_space.shape[0]
        action_highbound = env.action_space.high

        # print out parameters
        print('Parameters:')
        print('Observation Size=', observation_dim)
        print('Goal Size=', desired_goal_dim)
        print('State Size =', state_dim)
        print('Action Size =', action_dim)
        print('Action Upper Boundary =', action_highbound)

        self.replay_memory = Memory(int(args['memory_size']), int(args['seed']))
        # create actor
        self.actor = Actor(sess, state_dim, action_dim, action_highbound,
                      float(args['actor_lr']), float(args['tau']),
                      int(args['batch_size']), tuple(args['hidden_sizes']))

        # create critic
        self.critic = Critic(sess, state_dim, action_dim,
                        float(args['critic_lr']), float(args['tau']),
                        float(args['gamma']),
                        self.actor.n_actor_vars,
                        tuple(args['hidden_sizes']))

        # noise
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=float(0.2) * np.ones(action_dim))
        sess.run(tf.global_variables_initializer())

    def learn(self):
        # start to train when there's enough experience
        if self.replay_memory.size() > self.batch_size:
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_memory.sample_batch(self.batch_size)

            # find TD -- temporal difference
            # actor find target action
            a2_batch = self.actor.predict_target(s2_batch)

            # critic find target q
            q2_batch = self.critic.predict_target(s2_batch, a2_batch)

            # add a decay of q to reward if not done
            r_batch_discounted = []
            for k in range(self.batch_size):
                if d_batch[k]:
                    r_batch_discounted.append(r_batch[k])
                else:
                    r_batch_discounted.append(r_batch[k] + self.critic.gamma * q2_batch[k])

            # train critic with state, action, and reward
            pred_q, _ = self.critic.train(s_batch,
                                          a_batch,
                                          np.reshape(r_batch_discounted, (self.batch_size, 1)))

            # actor find action
            a_outs = self.actor.predict(s_batch)

            # get comment from critic
            comment_gradients = self.critic.get_comment_gradients(s_batch, a_outs)

            # train actor with state and the comment gradients
            self.actor.train(s_batch, comment_gradients[0])

            # Update target networks
            self.actor.update()
            self.critic.update()

    def test(self):

        pass

    def choose_action(self, state, test=False):
        a = self.actor.predict(np.reshape(state, (1, self.actor.state_dim)))
        if not test:
            a = a + self.actor_noise.get_noise()
        return a

    def random_action(self):
        return [np.random.uniform(low=-self.actor.action_bound, high=self.actor.action_bound, size=self.actor.action_dim)]

    def remember(self, state, state_next, action, reward, done):
        # add normal experience to memory -- i.e. experience w.r.t. desired goal
        self.replay_memory.add(np.reshape(state, (self.actor.state_dim,)), np.reshape(action, (self.actor.action_dim,)), reward, done,
                          np.reshape(state_next, (self.actor.state_dim,)))


    def rememberHER(self, s_prime, s_prime_next, achieved_goal, info, action, env):
        # add hindsight experience to memory -- i.e. experience w.r.t achieved goal
        substitute_goal = achieved_goal.copy()
        substitute_reward = env.compute_reward(achieved_goal, substitute_goal, info)
        self.replay_memory.add(np.reshape(s_prime, (self.actor.state_dim,)), np.reshape(action, (self.actor.action_dim,)),
                          substitute_reward, True, np.reshape(s_prime_next, (self.actor.state_dim,)))


    def load_checkpoint(self):
        print("...Loading checkpoint...")
        #self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        #self.saver.save(self.sess, self.checkpoint_file)

