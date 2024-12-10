# EX04

In this repository, 2D wave dynamics can be generated and learned with neural networks.


## Getting Started

To install the ex04 python package, first create an environment (e.g., using the package managing system [anaconda](https://docs.anaconda.com/anaconda/install/linux/), cd into it, and install the ex04 package via

```
conda create -n ex04 python=3.11 -y && conda activate ex04
pip install -e .
```

> [!IMPORTANT]  
> Change into the `ex04` directory, which will be considered as _root directory_ in the following, that is,
> ```
> cd src/ex04
> ```


## Data

From the `ex04` directory, issue the following commands to generate the data we need for this exercise.

```
python data/data_generation/wave_data_generator.py --sequence-length 101 --skip-rate 1 --dataset-name 32x32_slow --dataset-type train --n-samples 500
python data/data_generation/wave_data_generator.py --sequence-length 101 --skip-rate 1 --dataset-name 32x32_slow --dataset-type val --n-samples 150
python data/data_generation/wave_data_generator.py --sequence-length 201 --skip-rate 1 --dataset-name 32x32_slow --dataset-type test --n-samples 150

python data/data_generation/wave_data_generator.py --sequence-length 101 --skip-rate 3 --dataset-name 32x32_fast --dataset-type train --n-samples 500
python data/data_generation/wave_data_generator.py --sequence-length 101 --skip-rate 3 --dataset-name 32x32_fast --dataset-type val --n-samples 150
python data/data_generation/wave_data_generator.py --sequence-length 201 --skip-rate 3 --dataset-name 32x32_fast --dataset-type test --n-samples 150
```

> [!NOTE]
> The generated data must be sitting in the `data/numpy/` directory, for example `data/numpy/32x32_slow/train/*.npy`

> [!TIP]
> To visualize a single sample in an animation, use the `--visualize` flag when calling the data generation script, i.e.,
> ```
> python data/data_generation/wave_data_generator.py --visualize
> ```


## Training

Training and evaluation require the `ex04` environment activated, thus, if not done yet:
```
conda activate ex04
```

Model training can be invoked via the training script, e.g., by calling
```
python scripts/train.py model.name=my_clstm_model device="cpu"
```
to train a ConvLSTM to predict two-dimensional waves propagating slowly (since ```data.dataset_name=32x32_slow``` as default.

In contrast, a U-Net with channel sizes 2 and 4 in the first and second level and a context length of 4 (default is 1) on the fast waves can be trained on a GPU when calling
```
python scripts/train.py +experiment=unet model.name=unet_fast_ctxt4 model.c_list=[2,4] data.dataset_name=32x32_fast data.context_size=4 data.use_x_share_of_samples=0.1 device=cuda:0
```

Model checkpoints (holding the trained weights) and Tensorboard logs for inspecting the training progression are written to model-specific output directories, such as `outputs/unet_fast_ctxt4`.

> [!TIP]
> The progression of the training (convergence of error over training time) of one or multiple models can be inspected with Tensorboard by calling
> ```
> tensorboard --logdir outputs
> ```
> and opening `http://localhost:6006/` in a browser (e.g. Firefox).


## Evaluation

To evaluate a trained model, run
```
python scripts/evaluate.py -c outputs/unet_fast_ctxt4
```
The accurate model.name must be provided as -c argument (standing for "checkpoint"). The evaluation script will compute an RMSE and animate the wave predictions next to the ground truth.

Multi- and cross-model evaluations can be performed by passing multiple model names, e.g.,
```
python scripts/evaluate.py -c outputs/my_clstm_model outputs/unet_fast_ctxt4
```
Wildcards can be used to indicate a family of models by name, e.g.,
```
python scripts/evaluate.py -c outputs/*clstm*
```
evaluates all models in the `outputs` directory that have `clstm` in their name.

