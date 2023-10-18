import numpy as np
import matplotlib.pyplot as plt

class kArmedBandit:

    def __init__(self, arms: np.array):
        # define a k- armed badit (i.e. slot machine with k arms each with a unique distribution of rewards)

        self._arms = arms
        self._k = arms.shape[0]

    def pullArm(self, i: int) -> float:
        # pull the i-th arm of the bandit

        # returns reward for arm i (normally distributed with mean arm[i] and sd 1
        return np.random.normal(self._arms[i], 1)

    def armNum(self):
        return self._k

    def getArms(self):
        return self._arms


def eps_greedy(eps: float, bandit: kArmedBandit, steps: int, initVal: int, alpha: float):

    def take_step():
        if np.random.uniform(0, 1) < eps:
            action = np.random.randint(k)
            reward = bandit.pullArm(action)
            update(action, reward)
            temp = (((len(avg_reward) - 1) * avg_reward[-1]) + reward) / len(avg_reward)
            avg_reward.append(temp)

        else:
            # choose greedy action
            greed = np.max(Q)
            opt = np.where(Q == greed)[0]
            action = np.random.choice(opt)
            reward = bandit.pullArm(action)
            update(action, reward)
            temp = (((len(avg_reward) - 1) * avg_reward[-1]) + reward) / len(avg_reward)
            avg_reward.append(temp)

    def update(a: int, r: float):
        Q[a] = Q[a] + alpha * (r - Q[a])
        N[a] += 1

    avg_reward = [0.0]
    k = bandit.armNum()
    Q = np.full(shape=(k,), fill_value=initVal, dtype=float)
    N = np.ones(shape=(k,))
    for _ in range(steps):
        take_step()

    return avg_reward


def graph_results(averages: np.array, params: np.array):

    for i in range(averages.shape[0]):
        plt.plot(averages[i], label='eps = ' + str(params[i][0]) + ' Q_1 = ' + str(params[i][1]))

    plt.xlabel('Average Reward')
    plt.ylabel('Steps')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    k = 10
    arms = np.random.normal(0, 1, k)
    bandit = kArmedBandit(arms)

    params = np.array([[0.1, 0], [0, 5], [1, 0]])
    rewards = []

    for i in range(3):
        run_reward = eps_greedy(eps=params[i][0], bandit=bandit, steps=2000, initVal=params[i][1], alpha=0.5)
        rewards.append(run_reward)

    rewards = np.array(rewards)
    graph_results(rewards, params)
