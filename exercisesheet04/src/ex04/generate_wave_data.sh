#!/bin/bash

'''
# Generate Slow Wave Data
echo "Generating slow wave data (train)..."
python data/data_generation/wave_data_generator.py --sequence-length 101 --skip-rate 1 --dataset-name 32x32_slow --dataset-type train --n-samples 500

echo "Generating slow wave data (validation)..."
python data/data_generation/wave_data_generator.py --sequence-length 101 --skip-rate 1 --dataset-name 32x32_slow --dataset-type val --n-samples 150

echo "Generating slow wave data (test)..."
python data/data_generation/wave_data_generator.py --sequence-length 201 --skip-rate 1 --dataset-name 32x32_slow --dataset-type test --n-samples 150

# Generate Fast Wave Data
echo "Generating fast wave data (train)..."
python data/data_generation/wave_data_generator.py --sequence-length 101 --skip-rate 3 --dataset-name 32x32_fast --dataset-type train --n-samples 500

echo "Generating fast wave data (validation)..."
python data/data_generation/wave_data_generator.py --sequence-length 101 --skip-rate 3 --dataset-name 32x32_fast --dataset-type val --n-samples 150

echo "Generating fast wave data (test)..."
python data/data_generation/wave_data_generator.py --sequence-length 201 --skip-rate 3 --dataset-name 32x32_fast --dataset-type test --n-samples 150

# Optional: Visualize the Data
echo "Visualizing generated data..."
#python data/data_generation/wave_data_generator.py --sequence-length 101 --skip-rate 1 --visualize
echo "Visualizing generated data..."
python scripts/train.py model.name=my_clstm_model data.dataset_name=32x32_slow device="cpu"
echo "All tasks completed!"
'''



echo "train fast"
python scripts/train.py +experiment=unet model.name=unet_fast_ctxt4 data.dataset_name=32x32_fast device=cuda:0

echo "train slow"
python scripts/train.py +experiment=unet model.name=unet_slow data.dataset_name=32x32_slow device=cuda:0

echo "eval fast"
python scripts/evaluate.py -c outputs/unet_fast_ctxt4

echo "eval slow"
python scripts/evaluate.py -c outputs/unet_slow