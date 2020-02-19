import gym
import numpy
import tensorflow as tf


# Actor Class
class Actor(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, hidden_size):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_size = hidden_size

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
        f1 = 1. / numpy.sqrt(self.hidden_size[0])
        f2 = 1. / numpy.sqrt(self.hidden_size[1])
        f3 = 0.003
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

        # activation layer
        outputs = tf.layers.dense(x, units=self.action_dim, activation='tanh',
                                  kernel_initializer=tf.initializers.random_uniform(-f2, f2),
                                  bias_initializer=tf.initializers.random_uniform(-f2, f2))

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

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, n_actor_vars, hidden_size):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.hidden_size = hidden_size

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
        f1 = 1. / numpy.sqrt(self.hidden_size[0])
        f2 = 1. / numpy.sqrt(self.hidden_size[1])

        # state branch
        inputs = tf.placeholder(tf.float32, shape=[None, self.state_dim])
        x = tf.layers.dense(inputs, self.hidden_size[0],
                            kernel_initializer=tf.initializers.random_uniform(-f1, f1),
                            bias_initializer=tf.initializers.random_uniform(-f1, f1))
        x = tf.layers.batch_normalization(x)
        #x = tf.nn.relu(x)

        # action branch
        actions = tf.placeholder(tf.float32, shape=[None, self.action_dim])

        # merge
        x = tf.concat([x, actions], axis=1)
        x = tf.layers.dense(x, self.hidden_size[1],
                            kernel_initializer=tf.initializers.random_uniform(-f2, f2),
                            bias_initializer=tf.initializers.random_uniform(-f2, f2))
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)

        # activation layer
        f3 = 0.003
        outputs = tf.layers.dense(x, 1,
                                  kernel_initializer=tf.initializers.random_uniform(-f3, f3),
                                  bias_initializer=tf.initializers.random_uniform(-f3, f3))
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

    def __init__(self, environment, seed):

        self.actor
        self.critic
        self.noise
        # Load environment
        self.env = gym.make(environment)
        self.env.seed(int(seed))

        obs = self.env.reset()
        self.observation_dim = obs['observation'].shape[0]
        self.achieved_goal_dim = obs['achieved_goal'].shape[0]
        self.desired_goal_dim =  obs['desired_goal'].shape[0]

        pass

    def learn(self):
        pass

    def test(self):
        pass

    def load_models(self):
        pass

    def save_models(self):
        pass

    def choose_action(self):
        pass

    def remember(self):
        pass