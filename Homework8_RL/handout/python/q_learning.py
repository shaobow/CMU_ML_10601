import sys
import numpy as np
from environment import MountainCar


class LinearModel:
    def __init__(self, state_size: int, action_size: int, lr: float, indices: bool):
        """indices is True if indices are used as input for one-hot features.
        Otherwise, use the sparse representation of state as features
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.indices = indices
        self.w = np.zeros((state_size, action_size))
        self.b = 0

    def predict(self, state: dict[int, int]) -> list[float]:
        """
        Given state, makes predictions.
        """
        # form state dict into n-hot vector
        s = np.zeros((self.state_size, 1))
        for idx, hot in state.items():
            s[[idx], [0]] = hot
        # linear approximation (with bias) for all actions
        q = self.w.T @ s + self.b
        # list of q(s,a;w) for all actions
        q_list = q.reshape(self.action_size, ).tolist()
        # print(q_list)
        return q_list

    def update(self, state: dict[int, int], action: int, target: float):
        """
        Given state, action, and target, update weights.
        """
        # q value of given state and action based on current weight
        q_s_a = self.predict(state)[action]
        # form state dict into n-hot vector
        s = np.zeros((self.state_size, 1))
        for idx, hot in state.items():
            s[[idx], [0]] = hot
        # update weight for given action
        self.w[:, [action]] -= self.lr * (q_s_a - target) * s
        self.b -= self.lr * (q_s_a - target) * 1


class QLearningAgent:
    def __init__(self, env: MountainCar, mode: str = None, gamma: float = 0.9, lr: float = 0.01, epsilon: float = 0.05):
        """
        :param env: environment
        :param mode: raw or tile
        :param gamma: discount factor
        :param lr: learning rate
        :param epsilon: epsilon-greedy strategy
        """
        self.env = env
        self.mode = mode
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.linear_model = LinearModel(self.env.state_space, self.env.action_space,
                                        self.lr, indices=(self.mode == "tile"))

    def get_action(self, state: dict[int, int or float]) -> int:
        """epsilon-greedy strategy.
        Given state, returns action.
        """
        # print(state)
        rd = np.random.random_sample()
        # selects the optimal action with probability of 1-epsilon
        if rd < self.epsilon:
            action = np.random.randint(3)
        else:
            action = np.argmax(self.linear_model.predict(state))
        return action

    def train(self, episodes: int, max_iterations: int) -> list[float]:
        """training function.
        Train for 'episodes' iterations, where at most 'max_iterations` iterations
        should be run for each episode. Returns a list of returns.
        """
        returns = []
        for eps in range(episodes):
            _ = self.env.reset()
            reward_total = 0
            for itr in range(max_iterations):
                state = self.env.transform(self.env.state)
                action: int = self.get_action(state)
                next_state, reward, terminal = self.env.step(action)
                reward_total += reward
                target: float = reward + self.gamma*max(self.linear_model.predict(next_state))
                self.linear_model.update(state, action, target)
                if terminal:
                    break
            returns.append(reward_total)
            self.env.render()
        self.env.close()
        return returns


if __name__ == "__main__":
    # args = sys.argv
    # assert (len(args) == 9)
    # mode = args[1]
    # weight_out_dir = args[2]
    # returns_out_dir = args[3]
    # episodes = int(args[4])
    # iterations = int(args[5])
    # epsilon = float(args[6])
    # gamma = float(args[7])
    # lr = float(args[8])

    '''---------------debug------------------'''
    mode = "tile"
    weight_out_dir = "weight.out"
    returns_out_dir = "returns.out"
    episodes = 4000
    iterations = 200
    epsilon = 0.05
    gamma = 0.99
    lr = 0.01
    '''--------------------------------------'''

    # run parser and get parameters values
    '''remember to remove fixed initialization'''
    env = MountainCar(mode=mode)
    agent = QLearningAgent(env, mode=mode, gamma=gamma, epsilon=epsilon, lr=lr)
    returns_out = agent.train(episodes, iterations)

    bias = agent.linear_model.b
    weight = agent.linear_model.w
    weight_flat = weight.reshape((np.size(weight), 1))
    weight_out = np.concatenate(([[bias]], weight_flat), axis=0)

    np.savetxt(weight_out_dir, weight_out, delimiter='\n')
    np.savetxt(returns_out_dir, returns_out, delimiter='\n')

    # python q_learning.py tile fixed_weight.out fixed_returns.out 25 200 0.0 0.99 0.005
