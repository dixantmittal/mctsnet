import logging
from copy import deepcopy
from datetime import datetime
from itertools import count

import torch as t
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from device import Device
from environment import SIMULATOR
from expert import expert


def collector(idx, shared_model, shared_dataset, hyperparameters, lock):
    try:
        writer = SummaryWriter('runs/{}/collector:{:02}'.format(datetime.now().strftime("%d|%m_%H|%M"), idx))
        logging.basicConfig(filename='logs/collector:{:02}.log'.format(idx),
                            filemode='w',
                            format='%(message)s',
                            level=logging.DEBUG)

        # allocate a device
        n_gpu = t.cuda.device_count()
        if n_gpu > 0:
            Device.set_device(idx % n_gpu)

        local_model = deepcopy(shared_model)
        local_model.to(Device.get_device())
        local_model.eval()

        simulator = SIMULATOR()

        for itr in tqdm(count(), position=idx, desc='collector:{:02}'.format(idx)):
            local_model.load_state_dict(shared_model.state_dict())

            state = simulator.reset()

            episode_reward = 0
            for i in range(50):
                # Find the expert action for input belief
                expert_action, _ = expert(state, hyperparameters)

                lock.acquire()
                shared_dataset.append((state, expert_action))
                lock.release()

                # Simulate the learner's action
                action, _ = local_model.search(state, hyperparameters)
                state, reward, terminal = simulator.step(action)
                episode_reward += reward

                if terminal:
                    break

            logging.debug('Episode reward: {:.2f}'.format(episode_reward))
            writer.add_scalar('episode_reward', episode_reward, itr)
            writer.close()

    except KeyboardInterrupt:
        print('exiting collector:{:02}'.format(idx))
