#!/bin/bash
export PYTHONPATH=/mnt:$PYTHONPATH
python -m model.main -cfg config/ffhq-beta-vae.json --train-vae --scratch $1
