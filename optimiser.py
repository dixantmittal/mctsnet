import logging
from copy import deepcopy
from datetime import datetime
from itertools import count
from random import choice

import torch as t
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from device import Device
from calculate_loss import calculate_loss
from optimise_model import optimise_model


def optimiser(idx, shared_model, shared_dataset, hyperparameters, lock):
    try:
        writer = SummaryWriter('runs/{}/optimiser:{:02}'.format(datetime.now().strftime("%d|%m_%H|%M"), idx))
        logging.basicConfig(filename='logs/optimiser:{:02}.log'.format(idx),
                            filemode='w',
                            format='%(message)s',
                            level=logging.DEBUG)

        optimiser = t.optim.SGD(params=shared_model.parameters(), lr=hyperparameters.lr)

        # allocate a device
        n_gpu = t.cuda.device_count()
        if n_gpu > 0:
            Device.set_device(idx % n_gpu)

        local_model = deepcopy(shared_model)
        local_model.to(Device.get_device())
        local_model.train()

        for itr in tqdm(count(), position=idx, desc='optimiser:{:02}'.format(idx)):
            # Sync local model with shared model
            if itr % hyperparameters.sync_frequency == 0:
                local_model.load_state_dict(shared_model.state_dict())

            # Sample a data point from dataset
            state, expert_action = choice(shared_dataset)

            # Find the predicted action
            action, training_info = local_model.search(state, hyperparameters)

            # Optimise for the sample
            loss = calculate_loss(training_info, expert_action, hyperparameters)

            optimise_model(shared_model, local_model, loss, optimiser, lock)

            # Log the results
            logging.debug('Sample loss: {:.2f}'.format(loss.item()))
            writer.add_scalar('loss/sample_loss', loss.item(), itr)
            writer.close()

    except KeyboardInterrupt:
        print('exiting optimiser:{:02}'.format(idx))
