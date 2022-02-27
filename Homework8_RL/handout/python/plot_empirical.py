import numpy as np
from q_learning import QLearningAgent
from q_learning import MountainCar
import matplotlib.pyplot as plt


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concatenate((a[0:n-1] ,ret[n - 1:] / n))


fig = 1
for mode in ["raw", "tile"]:
    if mode == "raw":
        episodes = 2000
        iterations = 200
        epsilon = 0.05
        gamma = 0.999
        lr = 0.001
    elif mode == "tile":
        episodes = 400
        iterations = 200
        epsilon = 0.05
        gamma = 0.99
        lr = 0.00005

    # run parser and get parameters values
    '''remember to remove fixed initialization'''
    env = MountainCar(mode=mode)
    agent = QLearningAgent(env, mode=mode, gamma=gamma, epsilon=epsilon, lr=lr)
    returns_out = agent.train(episodes, iterations)

    returns = np.array(returns_out)
    returns_roll_mean = moving_average(returns, 25)
    # print(returns)
    # print(returns_roll_mean)
    plt.figure(fig, dpi=200)
    plt.plot(returns, label='return per episode')
    plt.plot(returns_roll_mean, label='rolling mean over window of 25')
    plt.legend()
    plt.title('Plot of ' + mode.capitalize())
    plt.xlabel('number of episodes')
    plt.ylabel('rewards value')
    plt.show()
    fig += 1
