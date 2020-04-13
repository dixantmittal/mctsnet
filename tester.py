import argparse

import torch as t

from device import Device
from solvers.mcts import MCTS
from solvers.mctsnet import MCTSnet
from utils.performer import performer

solvers = {'mctsnet': MCTSnet(),
           'mcts': MCTS()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', dest='solver', default='mctsnet', help='Solver to use')
    parser.add_argument('--load_model', dest='load_model', default='models/checkpoint.model', help='Path to load model file')
    parser.add_argument('--n_simulations', dest='n_simulations', default=10, type=int, help='Number of tree simulations')
    args = parser.parse_args()

    args.training = False

    if t.cuda.is_available():
        Device.set_device(0)

    model = solvers[args.solver]
    model.load(args.load_model)
    model.to(Device.get_device())

    with t.no_grad():
        print('Episode reward:', performer(model, args, render=True))
