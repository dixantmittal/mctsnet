#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3,4,5,6,7 \
python -W ignore:semaphore_tracker:UserWarning benchmarker.py \
  --solver=mctsnet \
  --load_model=models/rocksample/7_4_fixed_rocks_fixed_start/itr20.model \
  --n_simulations=20 \
  --n_workers=10 \
  --n_samples=100
