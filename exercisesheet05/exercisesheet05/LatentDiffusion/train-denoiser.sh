#!/bin/bash
export PYTHONPATH=/mnt:$PYTHONPATH
python -m model.main -cfg config/ffhq-denoiser.json --train-denoiser --load-vae vae.ckpt --scratch $1
