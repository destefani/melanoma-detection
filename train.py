import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from dataset import MelanomaDataset
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import wandb
wandb.init(project="melanoma-detection", entity="mdestefani")


root_dir = "dataset/siim-isic-melanoma-classification/jpeg/train/"
annotations_csv = "dataset/siim-isic-melanoma-classification/train.csv"

os.mkdir(f"results/{wandb.run.name}")


wandb.config.update({
    "epochs": 30,
    "batch_size": 32,
    "n_workers": 8,
    "learning_rate": 1e-3,
    "lambda_l2": 1e-5
})

# Transforms

train_transforms = transforms.Compose([

    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.8310, 0.6339, 0.5969],
        std=[0.1537, 0.1970, 0.2245]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=[0.8310, 0.6339, 0.5969],
        std=[0.1537, 0.1970, 0.2245])
])

# Create datasets

train_dataset = MelanomaDataset(
    annotations_csv,
    root_dir=root_dir,
    transform=train_transforms,
    test=False,
    test_size=0.2,
    seed=42)

test_dataset = MelanomaDataset(
    annotations_csv,
    root_dir=root_dir,
    transform=train_transforms,
    test=True,
    test_size=0.2,
    seed=42)


def get_sampler(labels_df):
    unique_labels, count = np.unique(labels_df, return_counts=True)
    label_weight = [sum(count) / c for c in count]
    weights = [label_weight[e] for e in labels_df]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

sampler = get_sampler(train_dataset.y_train)

train_loader = DataLoader(
    train_dataset,
    wandb.config.batch_size,
    num_workers=wandb.config.n_workers,
    sampler=sampler,
    pin_memory=True
    )

val_loader = DataLoader(
    test_dataset,
    wandb.config.batch_size,
    num_workers=wandb.config.n_workers,
    shuffle=True,
    pin_memory=True
    )

device = "cuda" if torch.cuda.is_available else "cpu"


model = torch.hub.load("pytorch/vision:v0.10.0", "resnet34", pretrained=True)
model.fc = nn.Linear(512, 1)
model.to(device)


criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=wandb.config.learning_rate,
    weight_decay=wandb.config.lambda_l2
)

for epoch in range(wandb.config.epochs):
    wandb.watch(model)
    # Training
    running_loss = 0.0
    correct_preds = 0.0
    wrong_preds = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).float()

        model.train()
        # 1. Feed forward to get logits
        y_pred = model(images)
        y_pred = torch.sigmoid(y_pred).view(-1)

        # 2. Compute loss and accuracy
        loss = criterion(y_pred, labels)
        correct_preds += (labels == torch.round(y_pred)).sum()
        wrong_preds += (labels != torch.round(y_pred)).sum()

        # 3. zero gradients before running the backward pass
        optimizer.zero_grad()

        # 4. Backward pass to compute the gradient of the loss
        # w.r.t our learnable parameters
        loss.backward()

        # 5. Update parameters
        optimizer.step()

        running_loss += loss.item()


    acc = correct_preds / (correct_preds + wrong_preds)
    wandb.log({
        "train_loss": running_loss,
        "train_acc": acc
    })
    print(f"Train - Epoch: {epoch}, Loss: {running_loss: .2f}, Accuracy: {acc: .4f}")

    # Validation
    running_loss = 0.0
    correct_preds = 0.0
    wrong_preds = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device).float()       

            model.eval()
            y_pred = model(images)
            y_pred = torch.sigmoid(y_pred).view(-1)
            loss = criterion(y_pred, labels)

            running_loss += loss.item()
            correct_preds += (labels == torch.round(y_pred)).sum()
            wrong_preds += (labels != torch.round(y_pred)).sum()

        acc = correct_preds / (correct_preds + wrong_preds)
        wandb.log({
            "val_loss": running_loss,
            "val_acc": acc
        })
        print(f"Eval  - Epoch: {epoch}, Loss: {running_loss: .2f}, Accuracy: {acc: .4f}")


    # Save model checpoint
    if (epoch + 1) % 10 == 0:
        torch.save(model, f"results/{wandb.run.name}/{epoch:04d}.pt")