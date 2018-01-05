import numpy as np
import math


class epsGreedyAgent:
    def __init__(self, A, epsilon):
        self.epsilon = epsilon
        self.A = A
        self.Q = np.zeros(A)
        self.numpick = np.zeros(A)

    def interact(self):
        rand = np.random.uniform()
        if rand < self.epsilon:
            a = np.random.randint(0, self.A)
        else:
            a = np.argmax(self.Q)
        return a

    def update(self, a, r):
        self.numpick[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.numpick[a]


class optimistEpsGreedyAgent(epsGreedyAgent):
    def __init__(self, A, epsilon, optimism):
        # TO IMPLEMENT
        self.epsilon = epsilon
        self.A = A
        self.Q = np.zeros(A) + 5 # optimistic initial value estimation
        self.numpick = np.zeros(A)


class softmaxAgent:
    def __init__(self, A, temperature):
        # TO IMPLEMENT
        self.A = A
        self.temperature = temperature
        self.Q = np.zeros(A)
        self.numpick = np.zeros(A)
        # max of value estimation, to prevent exp(Q) from explosion
        self.q_max = 0

    def interact(self):
        # TO IMPLEMENT
        powers = (self.Q - self.q_max)/self.temperature
        energy = np.exp(powers)
        probability = energy/np.sum(energy)
        action = np.random.choice(range(self.A), p=probability)
        return action

    def update(self, a, r):
        # TO IMPLEMENT
        self.numpick[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.numpick[a]
        if self.Q[a] > self.q_max:
            self.q_max = self.Q[a]


class UCBAgent:
    def __init__(self, A):
        # TO IMPLEMENT
        self.timestamp = 1
        self.A = A
        self.Q = np.zeros(A)
        self.numpick = np.zeros(A)

    def interact(self):
        # TO IMPLEMENT
        if np.min(self.numpick) == 0:
            return np.argmin(self.numpick)
        else:
            expectation = self.Q + np.sqrt(2 * np.log(self.timestamp) / self.numpick)
            return np.argmax(expectation)

    def update(self, a, r):
        # TO IMPLEMENT
        self.timestamp += 1
        self.numpick[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.numpick[a]


class ThompsonAgent:
    def __init__(self, A, mu_0, var_0):
        # TO IMPLEME
        self.A = A
        self.mu_est = np.zeros(A) + mu_0
        self.var_est = np.zeros(A) + var_0

    def interact(self):
        # TO IMPLEMENT
        q_est = np.sqrt(self.var_est) * np.random.randn(self.A) + self.mu_est
        action = np.argmax(q_est)
        return action

    def update(self, a, r):
        # TO IMPLEMENT
        # Update Bayesian prior on
        var0 = 1 # default variance of reward
        mu = self.mu_est[a]
        var = self.var_est[a]
        self.mu_est[a], self.var_est[a] = (r * var + mu * var0)/(var + var0), var * var0 / (var + var0)
