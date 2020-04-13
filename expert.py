from copy import deepcopy

from solvers.mcts import MCTS


def expert(state, args):
    args = deepcopy(args)
    args.n_simulations = 1000

    return MCTS.search(state, args)
