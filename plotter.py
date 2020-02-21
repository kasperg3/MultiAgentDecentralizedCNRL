import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib import style


class Plotter():
    def __init__(self, data_colors, legend_list, xlabel='xlabel', ylabel='ylabel', title='title', xlim=None, ylim=None):
        self.ylim = ylim
        self.xlim = xlim
        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.legend_list = legend_list
        self.data_colors = data_colors

    def plot(self, datapoints):
        for i in datapoints:
            plt.plot(i[0], i[1])


    def lillesvin(self):
        reward_list = []
        episode_list = []
        max_q_list = []
        plt.ion()
        fig = plt.figure()
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.title('Title')  # TODO: dynamically follow the settings that are experimented with
        p1, = plt.plot(episode_list, max_q_list, color='red', label='max_Q')
        p2, = plt.plot(episode_list, reward_list, color='blue', label='reward')
        plt.legend([p1, p2], ["max_Q", "reward"], loc=2)
        axes = plt.gca()
        axes.set_xlim([0, 2000])

        # Plot wrapper
        plt.plot(episode_list, max_q_list, color='red', label='max_Q')
        plt.plot(episode_list, reward_list, color='blue', label='reward')
        plt.show()
        plt.pause(0.05)

    def plotLearning(scores, filename, x=None, window=5):
        N = len(scores)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(scores[max(0, t - window):(t + 1)])
        if x is None:
            x = [i for i in range(N)]
        plt.ylabel('Score')
        plt.xlabel('Game')
        plt.plot(x, running_avg)
        plt.savefig(filename)



if __name__ == '__main__':
    plot_ = Plotter(['red', 'blue', 'green'], ['max_q', 'reward', 'penis'], 'xlabel', 'ylabel', 'title')
    pltter = Plotter()
    pltter.plotLearning()
    plot_.plot([[9, 1]
    #plot_.plot([[9, 1], [1, 2], [2, 3], [2, 4], [3, 5], [3, 6]])
    plt.show()
    plt.pause(0.05)


