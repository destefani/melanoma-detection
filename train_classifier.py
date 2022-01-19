import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import wandb

import monai
from monai.config import print_config
from monai.data import CacheDataset
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import CheckpointSaver, StatsHandler, ValidationHandler, ROCAUC
from monai.networks.nets import DenseNet121, DenseNet201

from monai.transforms import (
    AsChannelFirstd,
    Compose,
    CenterSpatialCropd,
    LoadImaged,
    RandAxisFlipd,
    RandRotated,
    RandFlipd,
    Rand2DElasticd,
    RandZoomd,
    RandZoom,
    ScaleIntensity,
    EnsureType,
    NormalizeIntensityd,
    Resized,
    ToTensord,
)
from monai.utils import set_determinism
from ignite.contrib.handlers.wandb_logger import *
from ignite.engine import Events

from training.data import prepare_data, weighted_sampler


print_config()

# Wandb config
wandb.init(project="melanoma-detection", entity="mdestefani")
results_dir = os.path.join("results", wandb.run.name)

ROOT_DIR = "dataset/siim-isic-melanoma-classification/jpeg/train"
LABELS_DF = "dataset/siim-isic-melanoma-classification/train.csv"

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


def main():
    wandb.config.update(
        {
            "epochs": 50,
            "batch_size": 128,
            "n_workers": 8,
            "learning_rate": 0.001,
            "lambda_l2": 0,
            "model": "densenet121",
            "pretrained": True,
            "warmup": True,
            "cache_rate": 0.05,
        }
    )

    print(wandb.config)

    train_transforms = Compose(
        [
            LoadImaged("image", image_only=True),
            AsChannelFirstd("image"),
            Resized("image", (256, 256)),
            CenterSpatialCropd("image", 224),
            NormalizeIntensityd("image"),
            RandRotated("image", range_x=15, prob=0.5, keep_size=True),
            RandFlipd("image", spatial_axis=0, prob=0.5),
            Rand2DElasticd("image", (0.3, 0.3), (1.0, 2.0)),
            RandZoomd("image", min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ToTensord("image"),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged("image", image_only=True),
            AsChannelFirstd("image"),
            Resized("image", (256, 256)),
            CenterSpatialCropd("image", 224),
            NormalizeIntensityd("image"),
            ToTensord("image"),
        ]
    )

    # Create datasets
    train_data, val_data = prepare_data(ROOT_DIR, LABELS_DF, warmup=wandb.config.warmup)

    train_ds = CacheDataset(
        data=train_data,
        transform=train_transforms,
        num_workers=8,
        cache_rate=wandb.config.cache_rate,
    )

    val_ds = CacheDataset(
        data=val_data,
        transform=val_transforms,
        num_workers=8,
        cache_rate=wandb.config.cache_rate,
    )

    # Manage class imbalance
    sampler = weighted_sampler(train_data)

    train_loader = DataLoader(
        train_ds,
        wandb.config.batch_size,
        num_workers=wandb.config.n_workers,
        sampler=sampler,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        wandb.config.batch_size,
        num_workers=wandb.config.n_workers,
        shuffle=True,
        pin_memory=True,
    )

    # Define network and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = DenseNet121(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        pretrained=wandb.config.pretrained,
    ).to(device)
    wandb.watch(model)

    criterion = nn.BCEWithLogitsLoss()  # sigmoid + binary cross entropy loss
    print(criterion)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=wandb.config.learning_rate,
        weight_decay=wandb.config.lambda_l2,
    )
    print(optimizer)

    iter_losses = []
    batch_sizes = []
    epoch_loss_values = []
    metric_values = []

    steps_per_epoch = len(train_ds) // train_loader.batch_size
    if len(train_ds) % train_loader.batch_size != 0:
        steps_per_epoch += 1

    def prepare_batch(batchdata, device, non_blocking):
        img, classes = batchdata["image"], batchdata["label"].unsqueeze(dim=1).float()
        return img.to(device), classes.to(device)

    def roc_auc_trans(x):
        if isinstance(x, list):
            pred = torch.cat([i[0][None, :] for i in x])
            label = torch.cat([i[1][None, :] for i in x])
            return pred, label
        return x["pred"], x["label"]

    val_handlers = [
        CheckpointSaver(
            save_dir=results_dir, save_dict={"model": model}, save_key_metric=True
        ),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        postprocessing=roc_auc_trans,
        key_val_metric={"roc_auc": ROCAUC(output_transform=roc_auc_trans)},
        prepare_batch=prepare_batch,
        decollate=False,
        val_handlers=val_handlers,
        amp=True if monai.utils.get_torch_version_tuple() >= (1, 6) else False,
    )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=wandb.config.epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=criterion,
        train_handlers=[ValidationHandler(1, evaluator)],
        prepare_batch=prepare_batch,
        amp=True if monai.utils.get_torch_version_tuple() >= (1, 6) else False,
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def _end_iter(engine):
        loss = np.average([o["loss"] for o in engine.state.output])
        batch_len = len(engine.state.batch[0])
        epoch = engine.state.epoch
        epoch_len = engine.state.max_epochs
        step = engine.state.iteration + 1
        iter_losses.append(loss)
        batch_sizes.append(batch_len)

        print(
            f"Train - epoch {epoch}/{epoch_len}, step {step}/{steps_per_epoch}, training_loss = {loss:.4f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        # the overall average loss must be weighted by batch size
        overall_average_loss = np.average(iter_losses, weights=batch_sizes)
        epoch_loss_values.append(overall_average_loss)

        # clear the contents of iter_losses and batch_sizes for the next epoch
        del iter_losses[:]
        del batch_sizes[:]

        # fetch and report the validation metrics
        roc = evaluator.state.metrics["roc_auc"]
        metric_values.append(roc)
        print(f"Eval- epoch {engine.state.epoch}, rocauc = {roc:.4f}")
        wandb.log(
            {
                "epoch": engine.state.epoch,
                "train_loss": overall_average_loss,
                "auc_score": roc,
            }
        )

    trainer.run()


if __name__ == "__main__":
    main()
