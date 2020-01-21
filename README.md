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
$ bash start_training.sh
```

##### Command-line arguments
Following arguments can be modified in the shell script or directly in the terminal.
  
| Argument         | Description                                                   | Values        | Default         |
|------------------|---------------------------------------------------------------|---------------|-----------------|
| --data           | File path containing the imitation data.                      | str           | None            |
| --save_model     | File path to save the trained model.                          | str           | ' '              |
| --load_model     | File path to load the trained model.                          | str           | ' '              |
| --n_simulations  | Name of the Simulator class to use for training.              | int           | 10              |
| --n_workers      | Number of workers for asynchronous update.                    | int           | cpu_count()     |
| --lr             | Learning rate for training.                                   | float         | 0.0005          |
| --epochs         | Number of epochs for training.                                | int           | 50              |
| --gamma          | Gamma value for geometric sum.                                | float (< 1.0) | 0.5             |
| --embedding_size | Size of the memory embedding.                                 | int           | 128             |


# Using different simulator
Make sure your custom simulator implements simulator.Base class. Set your new simulator as environment in the `environment.py`. Also, the MCTSnet model is trained using imitation learning. So, an expert policy is needed to generate the training data.

# Reference
Guez, Arthur, Théophane Weber, Ioannis Antonoglou, Karen Simonyan, Oriol Vinyals, Daan Wierstra, Rémi Munos, and David Silver. 2018. “Learning to Search with MCTSnets.” arXiv [cs.AI]. arXiv. http://arxiv.org/abs/1802.04697.
