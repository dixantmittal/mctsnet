#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3,4,5,6,7 \
python -W ignore:semaphore_tracker:UserWarning trainer.py \
  --data=data/rocksample/7_4_fixed_rocks_fixed_start.data \
  --load_model=models/rocksample/7_4_fixed_rocks_fixed_start/itr20.model \
  --save_model=models/rocksample/7_4_fixed_rocks_fixed_start/itr20.model \
  --n_simulations=20 \
  --n_optimisers=10 \
  --n_collectors=10 \
  --lr=0.0001 \
  --gamma=0.5 \
  --beta=0.001