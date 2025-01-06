#! /usr/bin/env python3

import os
import sys
import time
import threading
from torchsummary import summary

import hydra
import numpy as np
import torch as th
import torch.utils.tensorboard as tb

sys.path.append("")
from models import *
from data.pytorch_datasets import *
import utils.helpers as helpers


@hydra.main(config_path='../configs/', config_name='config', version_base=None)
def run_training(cfg):
    """
    Trains a model with the given configuration, printing progress to console and tensorboard and writing checkpoints
    to file.

    :param cfg: The hydra-configuration for the training
    """

    if cfg.seed:
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
    device = th.device(cfg.device)


    # Limit the number of threads to one (resulting in reasonable acceleration for CPU training)
    if cfg.limit_threads > 0: th.set_num_threads(cfg.limit_threads)

    # Initializing dataloaders for training and validation
    if cfg.verbose: print("\nInitialize datasets")
    train_dataset = hydra.utils.instantiate(config=cfg.data, mode="train")
    val_dataset = hydra.utils.instantiate(config=cfg.data, mode="val")
    train_dataloader = th.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    val_dataloader = th.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.validation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )


    # Set up model
    if cfg.verbose: print("\nInitialize model")
    model = hydra.utils.instantiate(config=cfg.model).to(device=device)
    def custom_model_summary(model):
        print(f"{'Layer':<40} {'Output Shape':<30} {'Param #':<20}")
        print("=" * 90)
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                total_params += num_params
                print(f"{name:<40} {str(tuple(param.shape)):<30} {num_params:<20}")
        print("=" * 90)
        print(f"Total trainable parameters: {total_params}")

    # Display model summary
    if cfg.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tModel {cfg.model.name} has {trainable_params} trainable parameters")
        print("\tModel Summary:")

        try:
            # Use sub-module summaries or a custom summary
            custom_model_summary(model)
        except Exception as e:
            print("\tUnable to generate model summary:", e)







    # Initialize training modules
    criterion = hydra.utils.instantiate(config=cfg.training.criterion)
    optimizer = hydra.utils.instantiate(config=cfg.training.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(config=cfg.training.scheduler, optimizer=optimizer)

    # Load checkpoint from file to continue training or initialize training scalars
    checkpoint_path = os.path.join("outputs", cfg.model.name, "checkpoints", f"{cfg.model.name}_last.ckpt")
    if cfg.training.continue_training:
        if cfg.verbose: print(f"\tRestoring model from {checkpoint_path}")
        checkpoint = th.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        iteration = checkpoint["iteration"]
        best_val_error = checkpoint["best_val_error"]
    else:
        epoch = 0
        iteration = 0
        best_val_error = np.inf

    # Write the model configurations to the model save path
    os.makedirs(os.path.join("outputs", cfg.model.name), exist_ok=True)

    # Initialize tensorbaord to track scalars
    writer = tb.SummaryWriter(log_dir=os.path.join("outputs", cfg.model.name, "tensorboard"))

    # Perform training by iterating over all epochs
    if cfg.verbose: print("\nStart training. Inspect progress via 'tensorboard --logdir outputs'")
    for epoch in range(epoch, cfg.training.epochs):

        # Track epoch and learning rate in tensorboard
        writer.add_scalar(tag="Epoch", scalar_value=epoch, global_step=iteration)
        writer.add_scalar(tag="Learning Rate", scalar_value=optimizer.state_dict()["param_groups"][0]["lr"],
                          global_step=iteration)

        start_time = time.time()

        # Train: iterate over all training samples
        outputs = list()
        targets = list()
        for inpt, trgt in train_dataloader:
            # Prepare inputs and targets
            inpt = inpt.to(device=device)
            trgt = trgt.to(device=device)
            # Perform optimization step and record outputs
            optimizer.zero_grad()
            otpt = model(
                x=inpt,
                teacher_forcing_steps=cfg.training.teacher_forcing_steps,
                epoch=epoch,
                target_len=len(trgt[0])
            )
            train_loss = criterion(otpt, trgt[:, :otpt.shape[1]])
            train_loss.backward()
            if cfg.training.clip_gradients:
                curr_lr = optimizer.param_groups[-1]["lr"] if scheduler is None else scheduler.get_last_lr()[0]
                th.nn.utils.clip_grad_norm_(model.parameters(), curr_lr)
            outputs.append(otpt.detach().cpu())
            targets.append(trgt.detach().cpu())
            optimizer.step()
            writer.add_scalar(tag="Loss/training", scalar_value=train_loss, global_step=iteration)
            iteration += 1
        with th.no_grad(): epoch_train_loss = criterion(th.cat(outputs), th.cat(targets)[:, :otpt.shape[1]]).numpy()

        # Validate (without gradients)
        with th.no_grad():
            outputs = list()
            targets = list()
            for inpt, trgt in val_dataloader:
                inpt = inpt.to(device=device)
                trgt = trgt.to(device=device)
                otpt = model(
                    x=inpt,
                    teacher_forcing_steps=cfg.validation.teacher_forcing_steps,
                    epoch=epoch,
                    target_len=len(trgt[0])
                )
                outputs.append(otpt.cpu())
                targets.append(trgt.cpu())
            epoch_val_loss = criterion(th.cat(outputs), th.cat(targets)[:, :otpt.shape[1]]).numpy()
        writer.add_scalar(tag="Loss/validation", scalar_value=epoch_val_loss, global_step=iteration)

        # Write model checkpoint to file, using a separate thread
        if cfg.training.save_model:
            if epoch_val_loss > best_val_error or epoch == cfg.training.epochs - 1:
                dst_path = checkpoint_path
            else:
                best_val_error = epoch_val_loss
                dst_path = f"{checkpoint_path.replace('last', 'best')}"
            thread = threading.Thread(
                target=helpers.write_checkpoint,
                args=(model, optimizer, scheduler, epoch, iteration, best_val_error, dst_path, ))
            thread.start()

        # Print training progress to console
        if cfg.verbose:
            epoch_time = round(time.time() - start_time, 2)
            print(f"Epoch {str(epoch).zfill(3)}/{str(cfg.training.epochs)}\t"
                  f"{epoch_time}s\t"
                  f"Train loss: {'%.2E' % epoch_train_loss}\t"
                  f"Val Loss: {'%.2E' % epoch_val_loss}")

        # Update learning rate
        if scheduler is not None: scheduler.step()

    # Wrap up by closing open threads
    try: thread.join(); writer.flush(); writer.close()
    except NameError: pass


if __name__ == "__main__":
    run_training()
    print("Done.")
