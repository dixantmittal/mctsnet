import torch as t

from device import Device
from environment import SIMULATOR
from utils.basic import to_one_hot


def prepare_input_for_f_backup(node, action, reward):
    memory = node.tensors.memory
    child_memory = node.variables.children[action].tensors.memory

    action = to_one_hot(action, SIMULATOR.n_actions).to(Device.get_device())
    reward = t.tensor([reward]).float().to(Device.get_device())

    return memory, child_memory, action, reward
