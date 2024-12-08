import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from config import Config
from utils.data_processor import DataSet
from models.vertebraenet import VertebraeNet
from utils.utils import save_model, load_model


# Training configurations
MAX_TRAIN_BATCHES = 10000
MAX_EVAL_BATCHES = 1000
ONE_CYCLE_MAX_LR = 0.0004
ONE_CYCLE_PCT_START = 0.3
SAVE_CHECKPOINT_EVERY_STEP = 500
N_FOLDS = 5


def evaluate(model, dataset, max_batches=1e9, shuffle=False):
    """
    Evaluate the VertebraeNet model on the evaluation dataset.
    """
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=os.cpu_count()
    )
    predictions, targets = [], []

    with torch.no_grad():
        for idx, (X, y) in enumerate(tqdm(dataloader, desc="Evaluating")):
            with autocast():
                y_pred = model.predict(X.to(Config.DEVICE))
            predictions.append(y_pred.cpu().numpy())
            targets.append(y.numpy())
            if idx >= max_batches:
                break

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    accuracy = np.mean(np.mean((predictions > 0.5) == targets, axis=0))
    return accuracy, predictions


def train_one_fold(fold, train_dataset, eval_dataset):
    """
    Train the VertebraeNet model on one fold.
    """
    print(f"Training fold {fold}...")

    # Initialize model, optimizer, scheduler, and scaler
    model = VertebraeNet().to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=ONE_CYCLE_MAX_LR,
        epochs=1,
        steps_per_epoch=min(MAX_TRAIN_BATCHES, len(train_dataset)),
        pct_start=ONE_CYCLE_PCT_START
    )
    scaler = GradScaler()

    # DataLoader for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count()
    )

    # Training loop
    model.train()
    for batch_idx, (X, y) in enumerate(tqdm(train_loader, desc=f"Training Fold {fold}")):
        optimizer.zero_grad()

        with autocast():
            predictions = model(X.to(Config.DEVICE))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, y.to(Config.DEVICE))

        # Gradient scaling and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Save intermediate checkpoint
        if batch_idx % SAVE_CHECKPOINT_EVERY_STEP == 0:
            save_model(f"vertebrae_fold_{fold}_step_{batch_idx}", model, optimizer, scheduler)

        if batch_idx >= MAX_TRAIN_BATCHES:
            break

    # Evaluate final model
    accuracy, _ = evaluate(model, eval_dataset, max_batches=MAX_EVAL_BATCHES)
    print(f"Fold {fold} - Final Accuracy: {accuracy:.4f}")

    # Save final model checkpoint
    save_model(f"vertebrae_fold_{fold}_final", model, optimizer, scheduler)
    return model


if __name__ == "__main__":
    # Load dataset
    print("Loading metadata...")
    df = pd.read_csv(f"{Config.METADATA_PATH}/meta_segmentation.csv")

    # K-Fold split
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, "fold"] = fold

    # Training loop for all folds
    for fold in range(N_FOLDS):
        print(f"Processing Fold {fold}...")

        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]

        train_dataset = DataSet(train_df, Config.TRAIN_IMAGES_PATH)
        eval_dataset = DataSet(val_df, Config.TRAIN_IMAGES_PATH)

        train_one_fold(fold, train_dataset, eval_dataset)

    print("Training completed for all folds.")