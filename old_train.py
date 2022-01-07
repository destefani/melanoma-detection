import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import wandb

from training.networks import ResNet
from training.dataset import ISIC2020


root_dir = "data/train/"
annotations_csv = "data/ISIC_2020_Training_GroundTruth_v2.csv"

# wand config
wandb.init(project="melanoma-detection", entity="mdestefani")
results_dir = os.path.join("results", wandb.run.name)

os.makedirs(results_dir)

wandb.config.update(
    {
        "epochs": 51,
        "batch_size": 16,
        "n_workers": 0,
        "learning_rate": 1e-3,
        "lambda_l2": 1e-5,
        "model": "resnet18"
    }
)

# Transforms
train_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.8310, 0.6339, 0.5969], std=[0.1537, 0.1970, 0.2245]
        ),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=[0.8310, 0.6339, 0.5969], std=[0.1537, 0.1970, 0.2245]
        ),
    ]
)

# Create datasets
train_dataset = ISIC2020(
    annotations_csv,
    root_dir=root_dir,
    transform=train_transforms,
    test=False,
    test_size=0.2,
    seed=42,
)

test_dataset = ISIC2020(
    annotations_csv,
    root_dir=root_dir,
    transform=train_transforms,
    test=True,
    test_size=0.2,
    seed=42,
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
    test_dataset,
    wandb.config.batch_size,
    num_workers=wandb.config.n_workers,
    shuffle=True,
    pin_memory=True,
)

device = "cuda" if torch.cuda.is_available else "cpu"

model = ResNet(wandb.config.model, pretrained=True, num_classes=1)
if torch.cuda.is_available():
    model.to(device)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=wandb.config.learning_rate,
    weight_decay=wandb.config.lambda_l2,
)

for epoch in range(wandb.config.epochs):
    wandb.watch(model)
    # Training
    train_loss = 0.0
    train_correct_preds = 0.0
    train_wrong_preds = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device).float()

        model.train()
        # 1. Feed forward to get logits
        y_pred = model(images)
        y_pred = torch.sigmoid(y_pred).view(-1)

        # 2. Compute loss and accuracy
        loss = criterion(y_pred, labels)
        train_correct_preds += (labels == torch.round(y_pred)).sum()
        train_wrong_preds += (labels != torch.round(y_pred)).sum()

        # 3. zero gradients before running the backward pass
        optimizer.zero_grad()

        # 4. Backward pass to compute the gradient of the loss
        # w.r.t our learnable parameters
        loss.backward()

        # 5. Update parameters
        optimizer.step()

        train_loss += loss.item()

    train_acc = train_correct_preds / (train_correct_preds + train_wrong_preds)
    print(f"Train - Epoch: {epoch}, Loss: {train_loss: .2f}, Accuracy: {train_acc: .4f}")

    # Validation
    val_loss = 0.0
    correct_preds = 0.0

    wrong_preds = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negatives = 0.0
    true_positive = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device).float()

            model.eval()
            y_pred = model(images)
            y_pred = torch.sigmoid(y_pred).view(-1)
            loss = criterion(y_pred, labels)
            try:
                tn, fp, fn, tp = confusion_matrix(
                    labels.cpu().numpy(), torch.round(y_pred).cpu().numpy()
                ).ravel()
                true_negative += tn
                false_positive += fp
                false_negatives += fn
                true_positive += tp
        
                auc_score = roc_auc_score(labels.cpu().numpy(), y_pred.cpu().numpy())    
            except:
                pass

            val_loss += loss.item()
            correct_preds += (labels == torch.round(y_pred)).sum()
            wrong_preds += (labels != torch.round(y_pred)).sum()

        val_acc = correct_preds / (correct_preds + wrong_preds)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "val_auc": auc_score
            }
        )
        print(
            f"Eval  - Epoch: {epoch}, Loss: {val_loss: .2f}, Accuracy: {val_acc: .4f}"
        )

    # Save model checpoint
    if (epoch) % 10 == 0:
        torch.save(model, f"results/{wandb.run.name}/{epoch:04d}.pt")