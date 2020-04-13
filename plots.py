import argparse
import json

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', required=True, help='Path to result data')
    parser.add_argument('--save_file', dest='save_file', required=True, help='Path to save plot')
    args = parser.parse_args()

    file = open(args.data, 'r')
    data = json.load(file)
    file.close()

    means = {}
    sem = {}
    time = {}
    for key in data:
        if key == 'environment':
            continue
        means[key] = [test['mean'] for test in data[key]]
        sem[key] = [test['stderr'] for test in data[key]]
        time[key] = [test['time'] for test in data[key]]

    for key in means:
        plt.errorbar(time[key], means[key], yerr=sem[key], label=key)

    plt.title(data['environment'])
    plt.ylabel('Mean Episode Rewards')
    plt.xlabel('Time per search (in seconds)')
    plt.ylim(0, 20)
    plt.legend()
    plt.savefig('{}.png'.format(args.save_file), dpi=1000)
