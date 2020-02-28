import os, sys, time
import json
import pickle

import gym
from gym import wrappers
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent import Actor, Critic, Agent
from replay import Memory
from agent import Noise
import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

# function to unpack observation from gym environment
def unpackObs(obs):
    return  obs['achieved_goal'], \
            obs['desired_goal'],\
            np.concatenate((obs['observation'], \
            obs['desired_goal'])), \
            np.concatenate((obs['observation'], \
            obs['achieved_goal']))


def find_low_high(dataset1, dataset2=None):
    highest_value = np.amax(dataset1)
    lowest_value = np.amin(dataset1)
    if dataset2 is not None:
        if highest_value < np.amax(dataset2):
            highest_value = np.amax(dataset2)
        if lowest_value < np.amax(dataset2):
            lowest_value = np.amin(dataset2)

    return lowest_value, highest_value


def normalize(dataset_1, dataset_2=None):
    dataset_low, dataset_high = find_low_high(dataset_1, dataset_2)
    dataset1 = np.divide(np.subtract(dataset_1, dataset_low), np.subtract(dataset_high, dataset_low))
    dataset2 = dataset_2
    if dataset_2 is not None:
        dataset2 = np.divide(np.subtract(dataset_2, dataset_low), np.subtract(dataset_high, dataset_low))
    return dataset1, dataset2



# Main
def main(args):
    # Set path to save result
    gym_dir = './' + args['env'] + '_' + args['variation'] + '/gym'

    # Set random seed for reproducibility
    np.random.seed(int(args['seed']))
    tf.set_random_seed(int(args['seed']))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Set random seed for reproducibility
    np.random.seed(int(args['seed']))
    tf.set_random_seed(int(args['seed']))

    # Load environment
    env = gym.make(args['env'])
    env.seed(int(args['seed']))

    sess = tf.Session()

    with sess:
        agent = Agent(args, sess, env=env)
        np.random.seed(0)

        model_dir = './' + args['env'] + '_' + args['variation'] + '/model'

        # initialize variables, create writer and saver
        saver = tf.train.Saver()

        try:
            saver.restore(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))
            print('Restore from previous training session')
        except:
            print('Start new training session')
        epoch_history = []
        epoch_test_history = []
        for k in range(args['epochs']):
            score_history = []
            for c in range(args['cycles']):
                for i in range(args['episodes']):
                    achieved_goal, desired_goal, state, state_prime = unpackObs(env.reset())
                    done = False
                    episode_score = 0
                    for j in range(int(args['episode_length'])):
                        if bool(np.random.binomial(1, 0.2)):
                            act = agent.random_action()
                        else:
                            act = agent.choose_action(state=state)

                        new_obs, reward, done, info = env.step(act[0])
                        achieved_goal, desired_goal, state_next, state_prime_next = unpackObs(new_obs)

                        # Store data in replay buffer
                        agent.remember(state, state_next, act[0], reward, done)
                        if bool(np.random.binomial(1, 0.8)):
                             agent.rememberHER(state_prime, state_prime_next, achieved_goal, info, act[0], env)

                        #Update the next state and add reward to episode_score
                        episode_score += reward
                        state = state_next

                        # render episode
                        if args['render']:
                            env.render()

                    #Save the episode scores
                    score_history.append(episode_score)
                    for t in range(int(args['optimizationsteps'])):
                        agent.learn()

                # TODO: Do rollout without random actions + noise, to see performance
                test_history = []
                for _l in range(int(args['rollouts'])):
                    achieved_goal, desired_goal, state, state_prime = unpackObs(env.reset())
                    done = False
                    episode_score = 0
                    for j in range(int(args['episode_length'])):
                        act = agent.choose_action(state=state, env=env, test=args['test'])
                        new_obs, reward, done, info = env.step(act[0])
                        achieved_goal, desired_goal, state_next, state_prime_next = unpackObs(new_obs)
                        episode_score += reward
                        state = state_next
                    test_history.append(episode_score)
                print('epoch:' + str(k) + " | score: %.2f | test score: %.2f " % (
                np.mean(score_history), np.mean(test_history)))

                # Save the histories of the epoch, both test and training
                epoch_test_history.append(test_history)
                epoch_history.append(score_history)
            # Take the mean of the scores and normalize it in the epoch and save it
            epoch_history.append(np.divide(score_history, int(args['episode_length'])))

            #Save model each epoch
            agent.save_checkpoint()
            saver.save(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))

    epoch_history_normalized, test_history_normalized = normalize(epoch_history, epoch_test_history)

    #plot the mean of the epochs
    epoch_plot, = plt.plot(range(args['epochs']), np.mean(epoch_history_normalized, axis=1), color='blue')
    test_plot, = plt.plot(range(args['epochs']), np.mean(test_history_normalized, axis=1), color='red')
    std_deviation = np.std(epoch_history_normalized, axis=1)
    std_deviation_test = np.std(test_history_normalized)
    plt.fill_between(range(args['epochs']),
                     np.subtract(np.mean(epoch_history_normalized, axis=1), std_deviation),
                     np.add(np.mean(epoch_history_normalized, axis=1), std_deviation), facecolor='blue', alpha=0.1)
    plt.fill_between(range(args['epochs']),
                     np.subtract(np.mean(test_history_normalized, axis=1), std_deviation_test),
                     np.add(np.mean(test_history_normalized, axis=1), std_deviation_test), facecolor='red', alpha=0.1)
    plt.ylabel('Reward')
    plt.xlabel('Epoch')
    plt.legend([epoch_plot, test_plot], [args['variation'], 'Test score'])
    plt.show()
    plt.pause(0.05)

    pickle.dump(epoch_plot, open("./plot_data/plot_data_" + args['variation'] + 'epoch', "wb"))
    pickle.dump(test_plot, open("./plot_data/plot_data_" + args['variation'] + 'test', "wb"))

    input("Press to exit")

    # close gym
    env.close()
    sess.close()

import gym_mergablerobots.envs.mergablerobots.ur_reach

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--actor-lr', help='actor learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic learning rate', default=0.001)
    parser.add_argument('--batch-size', help='batch size', default=256)
    parser.add_argument('--gamma', help='discount factor reward', default=0.99)
    parser.add_argument('--tau', help='target update tau', default=0.01)
    parser.add_argument('--memory-size', help='size of the replay memory', default=1000000)
    parser.add_argument('--hidden-sizes', help='number of nodes in hidden layer', default=(256, 256, 256))
    parser.add_argument('--epochs', help='number of epochs', default=50)
    parser.add_argument('--cycles', help='number of cycles to run in each epoch', default=50)
    parser.add_argument('--episodes', help='episodes to train in a cycle', default=16)
    parser.add_argument('--episode-length', help='max length of 1 episode', default=100)
    parser.add_argument('--optimizationsteps', help='number of optimization steps', default=40)
    parser.add_argument('--rollouts', help='Number of rollouts to run each epoch')
    # others and defaults
    parser.add_argument('--seed', help='random seed', default=1234)
    parser.add_argument('--render', help='render the gym env', action='store_true')
    parser.add_argument('--test', help='test mode does not do exploration', action='store_true')
    parser.add_argument('--variation', help='model variation name', default='DDPG_HER_TEST')
    #parser.set_defaults(env='FetchReach-v1')
    #parser.set_defaults(env='mergablerobots-v0')
    parser.set_defaults(env='UrReach-v0')
    #parser.set_defaults(env='FetchPickAndPlace-v1')
    parser.set_defaults(render=False)
    parser.set_defaults(test=False)

    # parse arguments
    args = vars(parser.parse_args())

    # run main
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
