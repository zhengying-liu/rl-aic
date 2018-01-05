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
        raise NotImplementedError


class softmaxAgent:
    def __init__(self, A, temperature):
        # TO IMPLEMENT
        raise NotImplementedError

    def interact(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def update(self, a, r):
        # TO IMPLEMENT
        raise NotImplementedError


class UCBAgent:
    def __init__(self, A):
        # TO IMPLEMENT
        raise NotImplementedError

    def interact(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def update(self, a, r):
        # TO IMPLEMENT
        raise NotImplementedError


class ThompsonAgent:
    def __init__(self, A, mu_0, var_0):
        # TO IMPLEME
        raise NotImplementedError

    def interact(self):
        # TO IMPLEMENT
        raise NotImplementedError

    def update(self, a, r):
        # TO IMPLEMENT
        raise NotImplementedError
