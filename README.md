# MCTSnet
An implementation of MCTSnet in PyTorch.

# Requirements
* Python 3

# Installation
```shell
$ python3 -m venv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```

# How to run
```shell
$ source .env/bin/activate
$ python main.py --<arg> <value>
```

##### Command-line arguments

| Argument         | Description                                                   | Values                          | Default         |
|------------------|---------------------------------------------------------------|---------------------------------|-----------------|
| --n_simulations  | Name of the Simulator class to use for training               | int.                            | 10              |
| --epochs         | Number of epochs for training                                 | int                             | 50              |
| --lr             | Learning rate for training                                    | float                           | 0.0005          |
| --batch_size     | Batch size for training                                       | int                             | 20              |
| --gamma          | Gamma value for geometric sum                                 | float (should be less than 1.0) | 0.5             |
| --alpha          | Strength of entropy regularisation                            | float                           | 1               |
| --embedding_size | Number of iterations for training [0 for infinite]            | int                             | 512             |


# Using different simulator
Make sure the simulator follows gym environment template. Edit the network structure for state shape and pass the simulator object to the tree. Also, the tree is trained using imitation learning. So, an expert policy is needed to generate the training data. Please follow the `main.py`, `simulator.py` and `networks.py` for the template

# Reference
Guez, Arthur, Théophane Weber, Ioannis Antonoglou, Karen Simonyan, Oriol Vinyals, Daan Wierstra, Rémi Munos, and David Silver. 2018. “Learning to Search with MCTSnets.” arXiv [cs.AI]. arXiv. http://arxiv.org/abs/1802.04697.
