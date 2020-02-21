import os, sys, time
import json
import gym
from gym import wrappers
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent import Actor, Critic, Agent
from replay import Memory
from agent import Noise
# function to unpack observation from gym environment
def unpackObs(obs):
    return  obs['achieved_goal'], \
            obs['desired_goal'],\
            np.concatenate((obs['observation'], \
            obs['desired_goal'])), \
            np.concatenate((obs['observation'], \
            obs['achieved_goal']))

# function to train agents
def train(sess, env, args, actor, critic, actor_noise, desired_goal_dim, achieved_goal_dim, observation_dim):

    # Set path to save results
    tensorboard_dir = './' + args['env'] + '_' + args['variation'] + '/train_' + datetime.now().strftime('%Y-%m-%d-%H')
    model_dir = './' + args['env'] + '_' + args['variation'] + '/model'

    # add summary to tensorboard

    # initialize variables, create writer and saver
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

    # restore session if exists
    try:
        saver.restore(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))
        print('Restore from previous training session')
    except:
        print('Start new training session')

    # initialize target network weights and replay memory
    actor.update()
    critic.update()
    replay_memory = Memory(int(args['memory_size']), int(args['seed']))

    # train in loop
    for i in range(int(args['episodes'])):

        # reset gym, get achieved_goal, desired_goal, state
        achieved_goal, desired_goal, s, s_prime = unpackObs(env.reset())
        episode_reward = 0
        episode_maximum_q = 0

        for j in range(int(args['episode_length'])):

            # render episode
            if args['render']:
                env.render()

            # predict action and add noise
            a = actor.predict(np.reshape(s, (1, actor.state_dim)))
            a = a + actor_noise.get_noise()

            # play
            obs_next, reward, done, info = env.step(a[0])
            achieved_goal, desired_goal, state_next, state_prime_next = unpackObs(obs_next)

            # add normal experience to memory -- i.e. experience w.r.t. desired goal
            replay_memory.add(np.reshape(s, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)), reward, done, np.reshape(state_next, (actor.state_dim,)))

            # add hindsight experience to memory -- i.e. experience w.r.t achieved goal
            substitute_goal = achieved_goal.copy()
            substitute_reward = env.compute_reward(achieved_goal, substitute_goal, info)
            replay_memory.add(np.reshape(s_prime, (actor.state_dim,)), np.reshape(a, (actor.action_dim,)), substitute_reward, True, np.reshape(state_prime_next, (actor.state_dim,)))

            # start to train when there's enough experience
            if replay_memory.size() > int(args['batch_size']):
                s_batch, a_batch, r_batch, d_batch, s2_batch = replay_memory.sample_batch(int(args['batch_size']))

                # find TD -- temporal difference
                # actor find target action
                a2_batch = actor.predict_target(s2_batch)

                # critic find target q
                q2_batch = critic.predict_target(s2_batch, a2_batch)

                # add a decay of q to reward if not done
                r_batch_discounted = []
                for k in range(int(args['batch_size'])):
                    if d_batch[k]:
                        r_batch_discounted.append(r_batch[k])
                    else:
                        r_batch_discounted.append(r_batch[k] + critic.gamma * q2_batch[k])

                # train critic with state, action, and reward
                pred_q, _ = critic.train(s_batch,
                                         a_batch,
                                         np.reshape(r_batch_discounted, (int(args['batch_size']), 1)))

                # record maximum q
                episode_maximum_q += np.amax(pred_q)

                # actor find action
                a_outs = actor.predict(s_batch)

                # get comment from critic
                comment_gradients = critic.get_comment_gradients(s_batch, a_outs)

                # train actor with state and the comment gradients
                actor.train(s_batch, comment_gradients[0])

                # Update target networks
                actor.update()
                critic.update()

            # record reward and move to next state
            episode_reward += reward
            s = state_next

            # if episode ends
            if done or j == int(args['episode_length'])-1:

                # print out results
                print('| Episode: {:d} | Reward: {:d} | Q: {:.4f}'.format(i, int(episode_reward),
                                                                          (episode_maximum_q / float(j))))
                # save model
                saver.save(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))

                break
    return

# function to test agents
def test(sess, env, args, actor, critic, desired_goal_dim, achieved_goal_dim, observation_dim):

    # Set path to save results
    tensorboard_dir = './' + args['env'] + '_' + args['variation'] + '/test_' + datetime.now().strftime('%Y-%m-%d-%H')
    model_dir = './' + args['env'] + '_' + args['variation'] + '/model'

    # add summary to tensorboard

    # initialize variables, create writer and saver
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

    # restore session 
    try:
        saver.restore(sess, os.path.join(model_dir, args['env'] + '_' + args['variation'] + '.ckpt'))
        print('Model is trained and ready')
    except:
        print('No model. Please train first. Exit the program')
        sys.exit()

    # test in loop
    for i in range(int(args['episodes'])):

        # reset gym, get achieved_goal, desired_goal, state
        achieved_goal, desired_goal, s, s_prime = unpackObs(env.reset())
        episode_reward = 0

        for j in range(int(args['episode_length'])):

            # render episode
            if args['render']:
                env.render()

            # predict action 
            a = actor.predict(np.reshape(s, (1, actor.state_dim)))

            # play
            obs_next, reward, done, info = env.step(a[0])
            achieved_goal, desired_goal, state_next, state_prime_next = unpackObs(obs_next)

            # record reward and move to next state
            episode_reward += reward
            s = state_next

            # if episode ends
            if done or j == int(args['episode_length']):
                # print out results
                print('| Episode: {:d} | Reward: {:d}'.format(i, int(episode_reward)))
                break
    return

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

        score_history = []

        for i in range(args['episodes']):
            achieved_goal, desired_goal, state, state_prime = unpackObs(env.reset())
            done = False
            episode_score = 0
            for j in range(int(args['episode_length'])):
                act = agent.choose_action(state=state)
                new_obs, reward, done, info = env.step(act[0])
                achieved_goal, desired_goal, state_next, state_prime_next = unpackObs(new_obs)

                # Store data in replay buffer
                agent.remember(state, state_next, act[0], reward, done)
                agent.rememberHER(state_prime, state_prime_next, achieved_goal, info, act[0], env)

                agent.learn()
                episode_score += reward

                #Update the next state
                state = state_next

                # render episode
                if args['render']:
                    env.render()
            score_history.append(episode_score)
            print('episode ', i, 'score %.2f' % episode_score,
                  'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
            if i % 25 == 0:
                agent.save_models()

    # close gym
    env.close()

    return

    with tf.Session() as sess:

        # Load environment
        env = gym.make(args['env'])
        env.seed(int(args['seed']))

        # get size of action and state (i.e. output and input for the agent)
        obs = env.reset()
        observation_dim = obs['observation'].shape[0]
        achieved_goal_dim = obs['achieved_goal'].shape[0]
        desired_goal_dim =  obs['desired_goal'].shape[0]
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

        # create actor
        actor = Actor(sess, state_dim, action_dim, action_highbound,
                      float(args['actor_lr']),
                      float(args['tau']),
                      int(args['batch_size']),
                      tuple(args['hidden_sizes']))

        # create critic
        critic = Critic(sess, state_dim, action_dim,
                        float(args['critic_lr']), float(args['tau']),
                        float(args['gamma']),
                        actor.n_actor_vars,
                        tuple(args['hidden_sizes']))

        # noise
        actor_noise = Noise(mu=np.zeros(action_dim))

        # train the network
        if not args['test']:
            train(sess, env, args, actor, critic, actor_noise, desired_goal_dim, achieved_goal_dim, observation_dim)
        else:
            test(sess, env, args, actor, critic, desired_goal_dim, achieved_goal_dim, observation_dim)

        # close gym
        env.close()

        # close session
        sess.close()

import gym_mergablerobots.envs.mergablerobots.ur_reach

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--actor-lr', help='actor learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic learning rate', default=0.001)
    parser.add_argument('--batch-size', help='batch size', default=64)
    parser.add_argument('--gamma', help='discount factor reward', default=0.99)
    parser.add_argument('--tau', help='target update tau', default=0.001)
    parser.add_argument('--memory-size', help='size of the replay memory', default=1000000)
    parser.add_argument('--hidden-sizes', help='number of nodes in hidden layer', default=(400, 300))
    parser.add_argument('--episodes', help='episodes to train', default=5000)
    parser.add_argument('--episode-length', help='max length of 1 episode', default=150)

    # others and defaults
    parser.add_argument('--seed', help='random seed', default=1234)
    parser.add_argument('--render', help='render the gym env', action='store_true')
    parser.add_argument('--test', help='test mode does not do exploration', action='store_true')
    parser.add_argument('--variation', help='model variation name', default='DDPG_HER')
    #parser.set_defaults(env='FetchReach-v1')
    parser.set_defaults(env='UrReach-v0')
    parser.set_defaults(render=False)
    parser.set_defaults(test=False)

    # parse arguments
    args = vars(parser.parse_args())

    # run main
    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
