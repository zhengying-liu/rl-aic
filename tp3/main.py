import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

from agents import epsGreedyAgent
from agents import optimistEpsGreedyAgent
from agents import softmaxAgent
from agents import ThompsonAgent
from agents import UCBAgent

parser = argparse.ArgumentParser(description='test bed for bandits algorithms')

subparsers = parser.add_subparsers(dest='agent')
subparsers.required = True

parser_eps = subparsers.add_parser('eps', description='epsilon greedy agent')
parser_opt = subparsers.add_parser('opt', description='optimistic epsilon'
                                   'greedy agent')
parser_soft = subparsers.add_parser('soft', description='softmax agent')
parser_ucb = subparsers.add_parser('ucb', description='ucb agent')
parser_thomps = subparsers.add_parser('thomps', description='thompson agent')

arg_dico = {'eps':'epsGreedyAgent', 'opt':'optimistEpsGreedyAgent',
            'soft':'softmaxAgent', 'ucb':'UCBAgent',
            'thomps':'ThompsonAgent'}

parser_eps.add_argument('--epsilon', type=float, metavar='e',
                        default=0.1, help='epsilon parameter')

parser_opt.add_argument('--epsilon', type=float, metavar='e',
                        default=0, help='epsilon parameter')
parser_opt.add_argument('--optimism', type=float, metavar='o',
                        default=5, help='optimism parameter')

parser_soft.add_argument('--temperature', type=float, metavar='T',
                         default=2e-1, help='temperature parameter')

parser_thomps.add_argument('--mu-0', type=float, metavar='M',
                           default=0, help='prior mean')
parser_thomps.add_argument('--var-0', type=float, metavar='V',
                           default=1, help='prior variance')

args = parser.parse_args()

agent = args.agent
agent_options = vars(args)
agent_options.pop('agent')

class Bandits:
    def __init__(self, N, k):
        self.cur = 0
        self.q_stars = np.random.randn(N, k)

    def select(self, n):
        self.cur = n

    def act(self, a):
        mean = self.q_stars[self.cur, a]
        reward = mean + np.random.randn()
        return reward

# Do not modify this class.

def plot_results(meanrewards, meanoptimals):
    plt.figure(1)
    plt.plot(meanrewards)
    plt.xlabel('Epoch')
    plt.ylabel('Average reward')
    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Percent optimal')
    plt.plot(meanoptimals)
    plt.show()

class AgentTester:
    def __init__(self, agentClass, N, k, iterations, params):
        self.iterations = iterations
        self.N = N
        self.agentClass = agentClass
        self.agentTable = []
        params['A'] = k
        for i in range(N):
            self.agentTable[len(self.agentTable):] = [agentClass(**params)]
        self.bandits = Bandits(self.N, k)
        self.optimal = np.argmax(self.bandits.q_stars, axis=1)

    def oneStep(self):
        rewards = np.zeros(self.N)
        optimals = np.zeros(self.N)
        for i in range(self.N):
            self.bandits.select(i)
            action = self.agentTable[i].interact()
            optimals[i] = (action == self.optimal[i]) and 1 or 0
            rewards[i] = self.bandits.act(action)
            self.agentTable[i].update(action, rewards[i])
        return rewards.mean(), optimals.mean() * 100

    def test(self):
        meanrewards = np.zeros(self.iterations)
        meanoptimals = np.zeros(self.iterations)
        try:
            for i in range(self.iterations):
                meanrewards[i], meanoptimals[i] = self.oneStep()
                display = '\nepoch: {:5.0f} -- mean reward: {:2.2f} -- ' + \
                          'percent optimal: {:2.1f}'
                sys.stdout.write(display.format(i, meanrewards[i], meanoptimals[i]))
                sys.stdout.flush()
            return meanrewards, meanoptimals
        except KeyboardInterrupt:
            print('\n\nInterupted, plotting results')
            last_fill = np.argwhere(meanrewards == 0)[0][0]
            return meanrewards[:last_fill], meanoptimals[:last_fill]

# Modify only the agent class and the parameter dictionnary.

if __name__ == '__main__':
    try:
        tester = AgentTester(eval(arg_dico[agent]), 2000, 10, 1000,
                             agent_options)
        # Do not modify.
        meanrewards, meanoptimals = tester.test()
        plot_results(meanrewards, meanoptimals)
    except NameError as e:
        print('Unimplemented agent: {}'.format(
              e.args[0]))

