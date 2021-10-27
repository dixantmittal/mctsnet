import argparse
import os
import shutil

import torch.multiprocessing as mp

from checkpoint import checkpoint
from collector import collector
from dataset import Dataset
from optimiser import optimiser
from solvers.mctsnet import MCTSnet

if __name__ == '__main__':
    mp.set_start_method('spawn', True)

    shutil.rmtree('runs', ignore_errors=True)
    if not os.path.exists('logs'):
        os.makedirs('logs')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, help='Path to imitation data')
    parser.add_argument('--save_model', default='', help='Path to save model file')
    parser.add_argument('--load_model', default='', help='Path to load model file')
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate for the training')
    parser.add_argument('--sync_frequency', default=5, type=int, help='Number of workers')
    parser.add_argument('--n_collectors', default=1, type=int, help='Number of workers')
    parser.add_argument('--n_optimisers', default=1, type=int, help='Number of workers')
    parser.add_argument('--n_simulations', default=10, type=int, help='Number of tree simulations')
    parser.add_argument('--gamma', default=0.5, type=float, help='Value for gamma')
    parser.add_argument('--beta', default=0.0001, type=float, help='Value for beta')
    args = parser.parse_args()

    args.training = True

    dataset = mp.Manager().list(Dataset(file=args.data))

    model = MCTSnet()
    model.load(args.load_model)
    model.share_memory()

    model_lock = mp.Lock()
    dataset_lock = mp.Lock()

    processes = [mp.Process(target=collector, args=(idx, model, dataset, args, dataset_lock)) for idx in range(args.n_collectors)]
    processes.extend([mp.Process(target=optimiser, args=(idx, model, dataset, args, model_lock)) for idx in range(args.n_collectors, args.n_collectors + args.n_optimisers)])
    processes.append(mp.Process(target=checkpoint, args=(model, dataset, args)))

    [p.start() for p in processes]

    try:
        [p.join() for p in processes]
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print('<< EXITING >>')
    finally:
        [p.terminate() for p in processes]

        os.system('clear')
        if input('Save model? (y/n): ') in ['y', 'Y', 'yes']:
            print('<< SAVING MODEL >>')
            model.save(args.save_model)
