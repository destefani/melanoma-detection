import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import pandas as pd
import numpy as np
import wandb

from monai.config import print_config
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    AddChannel,
    AsChannelFirstd,
    AsDiscrete,
    Compose,
    CenterSpatialCropd,
    LoadImaged,
    RandFlip,
    RandRotate,
    RandAxisFlipd,
    RandZoom,
    ScaleIntensity,
    EnsureType,
    NormalizeIntensityd,
    Resized,
    ToTensord
)
from monai.utils import set_determinism

from training.networks import ResNet, EfficientNet7
from training.dataset import ISIC2020, CacheISIC2020, PersistISIC2020

print_config()


root_dir = "dataset/isic/train/"
annotations_csv = "dataset/isic/ISIC_2020_Training_GroundTruth_v2.csv"

# wand config
wandb.init(project="melanoma-detection", entity="mdestefani")
results_dir = os.path.join("results", wandb.run.name)

os.makedirs(results_dir)

wandb.config.update(
    {
        "epochs": 70,
        "batch_size": 256,
        "n_workers": 8,
        "learning_rate": 0.002,
        "lambda_l2": 1e-5,
        "model": "efficientnetb7",
        "pretrained": True,
        "warmup": False,
    }
)

print(wandb.config)

# Transforms
train_transforms = Compose(
    [   
        LoadImaged("image", image_only=True),
        AsChannelFirstd("image"),
        Resized("image", (256, 256)),
        CenterSpatialCropd("image", 224),
        NormalizeIntensityd("image"),
        RandAxisFlipd("image", 0.5),
        ToTensord("image"),
    ])

val_transforms = Compose(
    [   
        LoadImaged("image", image_only=True),
        AsChannelFirstd("image"),
        Resized("image", (256, 256)),
        CenterSpatialCropd("image", 224),
        NormalizeIntensityd("image"),
        ToTensord("image"),
    ])

# Create datasets
train_dataset = CacheISIC2020(
    annotations_csv,
    root_dir=root_dir,
    transform=train_transforms,
    test=False,
    test_size=0.1,
    seed=42,
    warmup=wandb.config.warmup
)

val_dataset = CacheISIC2020(
    annotations_csv,
    root_dir=root_dir,
    transform=val_transforms,
    test=True,
    test_size=0.2,
    seed=42,
    warmup=wandb.config.warmup
)


def get_sampler(labels_df):
    unique_labels, count = np.unique(labels_df, return_counts=True)
    label_weight = [sum(count) / c for c in count]
    weights = [label_weight[e] for e in labels_df]
    return WeightedRandomSampler(weights, len(weights))


sampler = get_sampler(train_dataset.y_train)

train_loader = DataLoader(
    train_dataset,
    wandb.config.batch_size,
    num_workers=wandb.config.n_workers,
    sampler=sampler,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    wandb.config.batch_size,
    num_workers=wandb.config.n_workers,
    shuffle=True,
    pin_memory=True,
)

# Define network and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = EfficientNet7(pretrained=wandb.config.pretrained, num_classes=1)
wandb.watch(model)

if torch.cuda.is_available():
    model.to(device)
    print(model)

criterion = nn.BCELoss()
print(criterion)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=wandb.config.learning_rate,
    weight_decay=wandb.config.lambda_l2,
)
print(optimizer)

auc_metric = ROCAUCMetric()


for epoch in range(wandb.config.epochs):
    # Training
    model.train()
    train_loss = 0.0

    for batch_data in train_loader:
        images, labels = batch_data["image"].to(device), batch_data["target"].to(device).float()
        # 1. Feed forward to get logits
        y_pred = model(images).view(-1)
        y_pred = torch.sigmoid(y_pred)
        # 2. Compute loss and accuracy
        loss = criterion(y_pred, labels)
        # 3. zero gradients before running the backward pass
        optimizer.zero_grad()
        # 4. Backward pass to compute the gradient of the loss
        # w.r.t our learnable parameters
        loss.backward()
        # 5. Update parameters
        optimizer.step()
        train_loss += loss.item()  # check loss

    print(f"Train - Epoch: {epoch}, Loss: {train_loss:.4f}")

    # Validation
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.float32, device=device)
        for val_data in val_loader:
            images, labels = val_data["image"].to(device), val_data["target"].to(device).float()
            preds = model(images).view(-1)
            preds = torch.sigmoid(preds)
            y_pred = torch.cat([y_pred, preds], dim=0)
            y = torch.cat([y, labels], dim=0)
        
        val_loss += criterion(preds, labels).item()
        auc_metric(y_pred, y)
        result = auc_metric.aggregate()
        auc_metric.reset()

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "auc_score": result
            }
        )
        print(
            f"Eval  - Epoch: {epoch}, Loss: {val_loss:.4f}, AUC: {result:.4f}"
        )

    # Save model checpoint
    if (epoch) % 10 == 0:
        torch.save(model, f"results/{wandb.run.name}/{epoch:04d}.pt")