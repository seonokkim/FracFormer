import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from utils.data_processor import DataSet
from utils.utils import save_model, load_model, filter_nones, gc_collect
from models.vertebraenet import VertebraeNet
from models.fracturenet import FractureNet

def weighted_loss(y_pred_logit, y, reduction='mean'):
    """
    Custom weighted loss function to handle class imbalance.
    """
    neg_weights = torch.tensor([7., 1, 1, 1, 1, 1, 1]).to(Config.DEVICE)
    pos_weights = torch.tensor([14., 2, 2, 2, 2, 2, 2]).to(Config.DEVICE)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred_logit, y, reduction='none')
    weights = y * pos_weights + (1 - y) * neg_weights
    loss *= weights

    norm = torch.sum(weights, dim=1, keepdim=True)
    loss /= norm

    if reduction == 'mean':
        return loss.mean()
    return loss


def train_vertebraenet(df_train, logger=None):
    """
    Train VertebraeNet model for vertebra visibility prediction.
    """
    vertebrae_models = []

    for fold in range(Config.N_FOLDS):
        print(f"Training VertebraeNet for fold {fold}...")
        gc_collect()

        # Split data into train and validation sets
        ds_train = DataSet(df_train.query('split != @fold'), Config.TRAIN_IMAGES_PATH, transforms=None)
        ds_eval = DataSet(df_train.query('split == @fold'), Config.TRAIN_IMAGES_PATH, transforms=None)

        # Initialize model, optimizer, and scheduler
        model = VertebraeNet().to(Config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=Config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=Config.ONE_CYCLE_MAX_LR, epochs=1,
            steps_per_epoch=min(Config.MAX_TRAIN_BATCHES, len(ds_train)),
            pct_start=Config.ONE_CYCLE_PCT_START
        )

        model.train()
        scaler = torch.cuda.amp.GradScaler()

        for batch_idx, (X, y_vert) in enumerate(tqdm(DataLoader(ds_train, batch_size=Config.BATCH_SIZE, collate_fn=filter_nones), desc="Training VertebraeNet")):
            if batch_idx >= Config.MAX_TRAIN_BATCHES:
                break

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                y_vert_pred = model(X.to(Config.DEVICE))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(Config.DEVICE))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if batch_idx % Config.SAVE_CHECKPOINT_EVERY_STEP == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item()}")
                save_model(f"vertebraenet_fold{fold}", model)

        vertebrae_models.append(model)

    return vertebrae_models


def train_fracturenet(df_train, vertebrae_models, logger=None):
    """
    Train FractureNet model for fracture prediction using VertebraeNet predictions.
    """
    fracture_models = []

    for fold in range(Config.N_FOLDS):
        print(f"Training FractureNet for fold {fold}...")
        gc_collect()

        # Split data into train and validation sets
        ds_train = DataSet(df_train.query('split != @fold'), Config.TRAIN_IMAGES_PATH, transforms=None)
        ds_eval = DataSet(df_train.query('split == @fold'), Config.TRAIN_IMAGES_PATH, transforms=None)

        # Initialize model, optimizer, and scheduler
        model = FractureNet().to(Config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=Config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=Config.ONE_CYCLE_MAX_LR, epochs=1,
            steps_per_epoch=min(Config.MAX_TRAIN_BATCHES, len(ds_train)),
            pct_start=Config.ONE_CYCLE_PCT_START
        )

        model.train()
        scaler = torch.cuda.amp.GradScaler()

        for batch_idx, (X, y_frac, y_vert) in enumerate(tqdm(DataLoader(ds_train, batch_size=Config.BATCH_SIZE, collate_fn=filter_nones), desc="Training FractureNet")):
            if batch_idx >= Config.MAX_TRAIN_BATCHES:
                break

            optimizer.zero_grad()

            # Get VertebraeNet predictions to mask fracture labels
            vert_predictions = vertebrae_models[fold].predict(X.to(Config.DEVICE)).detach()

            # Mask fracture labels
            y_frac = y_frac * vert_predictions

            with torch.cuda.amp.autocast():
                y_frac_pred = model(X.to(Config.DEVICE))
                loss = weighted_loss(y_frac_pred, y_frac.to(Config.DEVICE))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if batch_idx % Config.SAVE_CHECKPOINT_EVERY_STEP == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item()}")
                save_model(f"fracturenet_fold{fold}", model)

        fracture_models.append(model)

    return fracture_models


if __name__ == "__main__":
    # Load training data
    print("Loading training data...")
    df_train = pd.read_csv(f"{Config.DATA_PATH}/train.csv")

    # Train VertebraeNet
    print("Training VertebraeNet...")
    vertebrae_models = train_vertebraenet(df_train)

    # Train FractureNet using VertebraeNet predictions
    print("Training FractureNet...")
    fracture_models = train_fracturenet(df_train, vertebrae_models)

    print("Training completed.")
