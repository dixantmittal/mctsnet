python -W ignore:semaphore_tracker:UserWarning trainer.py \
  --data=data/grid.dat \
  --save_model= \
  --load_model= \
  --n_simulations=20 \
  --n_workers=8 \
  --lr=0.0005 \
  --epochs=10 \
  --gamma=0.9 \
  --embedding_size=128
