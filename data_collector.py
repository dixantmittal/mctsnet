import argparse
import glob
import math
import os
import pickle

import torch.multiprocessing as mp
from tqdm import tqdm

from environment import SIMULATOR
from solvers.mcts import MCTS


def expert_policy(idx, n_samples, args):
    data = []

    my_simulator = SIMULATOR()

    progress = tqdm(range(n_samples), position=idx, desc='worker_{:02}'.format(idx))
    while len(data) < n_samples:
        state = my_simulator.reset()

        root = None
        for e in range(50):
            action, root = MCTS.search(state, args, root=root)

            data.append((state, action))

            state, reward, terminal = my_simulator.step(action)

            root = root.children[action]

            if terminal:
                break

            progress.update(1)

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    file = open('{}/{:02}.data'.format(args.dir, idx), 'wb')
    pickle.dump(data, file)
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--n_workers', default=1, type=int)
    parser.add_argument('--n_simulations', default=1000, type=int, help='Number of tree simulations')
    parser.add_argument('--n_samples', default=1000, type=int)
    parser.add_argument('--start_idx', default=0, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    args.dir = args.dir.rstrip('/')

    samples_per_worker = math.ceil(args.n_samples / args.n_workers)

    processes = [mp.Process(target=expert_policy, args=(idx, samples_per_worker, args)) for idx in range(args.start_idx, args.start_idx + args.n_workers)]

    [p.start() for p in processes]
    [p.join() for p in processes]

    data = []

    files = glob.glob('{}/*'.format(args.dir))
    for file in tqdm(files):
        file = open(file, 'rb')
        data.extend(pickle.load(file))
        file.close()

    file = open('{}.data'.format(args.dir.rstrip('/')), 'wb')
    pickle.dump(data, file)
    file.close()
