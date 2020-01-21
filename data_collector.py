import argparse
import math
import multiprocessing as mp
import pickle
from itertools import repeat

from tqdm import tqdm

from environment import ENVIRONMENT
from solvers.mcts import MCTS


def expert_policy(idx, n_samples):
    X, y = [], []

    my_simulator = ENVIRONMENT()
    for itr in tqdm(range(n_samples), position=idx, desc='worker_{:02}'.format(idx)):
        state = my_simulator.reset()

        root = None
        terminal = False
        while not terminal:
            action, root = MCTS.search(state, 1000, root)

            X.append(state)
            y.append(action)

            state, reward, terminal = my_simulator.step(action)

            root = root.children[action]

    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_file', dest='save_file', required=True)
    parser.add_argument('--n_workers', dest='n_workers', default=mp.cpu_count(), type=int)
    parser.add_argument('--n_samples', dest='n_samples', default=1000, type=int)
    args = parser.parse_args()

    n_cpus = args.n_workers
    samples_per_core = math.ceil(args.n_samples / n_cpus)

    data = mp.Pool(n_cpus).starmap_async(expert_policy, zip(range(n_cpus), repeat(samples_per_core))).get()

    X, y = [], []

    for data_point in data:
        X.extend(data_point[0])
        y.extend(data_point[1])

    file = open(args.save_file, 'wb')
    pickle.dump({'X': X, 'y': y}, file)
    file.close()
