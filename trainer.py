import argparse
from os import cpu_count

from dataset import StateDataset
from modules import Readout, Backup, Memory, Policy
from optimise import optimise
from solvers.mctsnet import MCTSnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', required=True, help='Path to imitation data')
    parser.add_argument('--save_model', dest='save_model', default='', help='Path to save model file')
    parser.add_argument('--load_model', dest='load_model', default='', help='Path to load model file')
    parser.add_argument('--n_simulations', dest='n_simulations', default=10, type=int, help='Number of tree simulations')
    parser.add_argument('--n_workers', dest='n_workers', default=cpu_count(), type=int, help='Number of workers')
    parser.add_argument('--lr', dest='lr', default=0.0005, type=float, help='Learning rate for the training')
    parser.add_argument('--epochs', dest='epochs', default=50, type=int, help='Number of epochs for training')
    parser.add_argument('--gamma', dest='gamma', default=0.5, type=float, help='Value for gamma')
    parser.add_argument('--embedding_size', dest='embedding_size', default=128, type=int, help='Size of Embedding')
    args = parser.parse_args()

    model = MCTSnet(f_memory=Memory(args.embedding_size),
                    f_policy=Policy(args.embedding_size),
                    f_backup=Backup(args.embedding_size),
                    f_readout=Readout(args.embedding_size))
    model.load(args.load_model)

    dataset = StateDataset(file=args.data)

    try:
        optimise(model, dataset, args)
    except KeyboardInterrupt:
        print('--EXITING--')
    except Exception as e:
        print(e)
    finally:
        save_model = input('Save model? (y/n): ')
        if save_model in ['y', 'Y', 'yes']:
            model.save(args.save_model)
