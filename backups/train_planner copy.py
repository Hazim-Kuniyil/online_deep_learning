"""
Usage:
    python3 -m homework.train_planner --exp_dir logs/planner --model_name mlp_planner --num_epoch 100 --lr 1e-3 --batch_size 32 --seed 42 --transform_pipeline default --num_workers 4
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

from .models import load_model, save_model, MODEL_FACTORY
from .metrics import PlannerMetric
from .datasets.road_dataset import load_data


def train_planner(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
    transform_pipeline: str = "default",
    num_workers: int = 4,
    **kwargs,
):
    """
    Trains the MLPPlanner model for waypoint prediction.

    Args:
        exp_dir (str): Directory to save logs and model checkpoints.
        model_name (str): Name of the model to train (e.g., 'mlp_planner').
        num_epoch (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.
        transform_pipeline (str): Data augmentation pipeline ('default' or 'aug').
        num_workers (int): Number of workers for data loading.
        **kwargs: Additional keyword arguments for model initialization.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using CUDA for training.")
    else:
        print("CUDA not available, using CPU.")

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    log_dir = Path(exp_dir) / f"{model_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")

    # Initialize the model
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    print(f"Initialized model: {model_name}")

    # Define the loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Define the optimizer (Adam)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    print("Defined loss function and optimizer.")

    # Load training and validation data
    train_data = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_data = load_data(
        dataset_path="drive_data/val",
        transform_pipeline="default",  # Use default for validation
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print("Loaded training and validation data.")

    # Initialize metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    global_step = 0
    best_val_l1_error = float('inf')  # Lower is better

    for epoch in range(1, num_epoch + 1):
        print(f"\nEpoch {epoch}/{num_epoch}")
        model.train()
        train_metric.reset()

        # Training Loop
        for batch_idx, batch in enumerate(train_data):
            track_left = batch['track_left'].to(device)        # Shape: (B, 10, 2)
            track_right = batch['track_right'].to(device)      # Shape: (B, 10, 2)
            waypoints_gt = batch['waypoints'].to(device)       # Shape: (B, 3, 2)
            waypoints_mask = batch['waypoints_mask'].to(device)  # Shape: (B, 3)

            # Forward pass
            waypoints_pred = model(track_left, track_right)    # Shape: (B, 3, 2)

            # Separate the components
            longitudinal_pred = waypoints_pred[:, :, 0]
            lateral_pred = waypoints_pred[:, :, 1]

            longitudinal_gt = waypoints_gt[:, :, 0]
            lateral_gt = waypoints_gt[:, :, 1]

            # Compute separate losses
            longitudinal_loss = criterion(longitudinal_pred * waypoints_mask, longitudinal_gt * waypoints_mask)
            lateral_loss = criterion(lateral_pred * waypoints_mask, lateral_gt * waypoints_mask)

            # Total loss
            loss = longitudinal_loss + 4 * lateral_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_metric.add(preds=waypoints_pred, labels=waypoints_gt, labels_mask=waypoints_mask)

            # Logging
            logger.add_scalar('Train/Loss', loss.item(), global_step)
            logger.add_scalar('Train/Loss_Longitudinal', longitudinal_loss.item(), global_step)
            logger.add_scalar('Train/Loss_Lateral', lateral_loss.item(), global_step)
            logger.flush()

            if global_step % 100 == 0:
                current_metrics = train_metric.compute()
                print(f"Step {global_step}: Total_Loss={loss.item():.4f}, "
                    f"Longitudinal_Loss={longitudinal_loss.item():.4f}, "
                    f"Lateral_Loss={lateral_loss.item():.4f}, "
                    f"L1_Error={current_metrics['l1_error']:.4f}")
                logger.flush()

            global_step += 1

        # Compute training metrics after epoch
        train_metrics = train_metric.compute()
        logger.add_scalar('Epoch/Train_L1_Error', train_metrics['l1_error'], epoch)
        logger.add_scalar('Epoch/Train_Longitudinal_Error', train_metrics['longitudinal_error'], epoch)
        logger.add_scalar('Epoch/Train_Lateral_Error', train_metrics['lateral_error'], epoch)
        logger.add_scalar('Train/Loss_Longitudinal', longitudinal_loss.item(), global_step)
        logger.add_scalar('Train/Loss_Lateral', lateral_loss.item(), global_step)
        print(f"Training Metrics: L1_Error={train_metrics['l1_error']:.4f}, "
              f"Longitudinal_Error={train_metrics['longitudinal_error']:.4f}, "
              f"Lateral_Error={train_metrics['lateral_error']:.4f}")

        # Validation Loop
        model.eval()
        val_metric.reset()
        with torch.no_grad():
            for batch in val_data:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints_gt = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)

                waypoints_pred = model(track_left, track_right)

                val_metric.add(preds=waypoints_pred, labels=waypoints_gt, labels_mask=waypoints_mask)

        # Compute validation metrics after epoch
        val_metrics = val_metric.compute()
        logger.add_scalar('Epoch/Val_L1_Error', val_metrics['l1_error'], epoch)
        logger.add_scalar('Epoch/Val_Longitudinal_Error', val_metrics['longitudinal_error'], epoch)
        logger.add_scalar('Epoch/Val_Lateral_Error', val_metrics['lateral_error'], epoch)
        print(f"Validation Metrics: L1_Error={val_metrics['l1_error']:.4f}, "
              f"Longitudinal_Error={val_metrics['longitudinal_error']:.4f}, "
              f"Lateral_Error={val_metrics['lateral_error']:.4f}")

        # Save the best model based on validation L1 error
        if val_metrics['l1_error'] < best_val_l1_error:
            best_val_l1_error = val_metrics['l1_error']
            best_model_path = log_dir / f"best_model_epoch_{epoch}.th"
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch} with L1_Error={best_val_l1_error:.4f}")

        # Save checkpoints every 10 epochs and the final epoch
        if epoch % 10 == 0 or epoch == num_epoch:
            checkpoint_path = log_dir / f"epoch_{epoch}.th"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save the final model using the provided save_model function
    final_model_path = save_model(model)
    print(f"\nFinal model saved to {final_model_path}")

    # Optionally, save the final state_dict in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}_final.th")
    print(f"Final model state_dict saved to {log_dir / f'{model_name}_final.th'}")

    # Close the logger
    logger.close()
    print("Training completed and logged successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLPPlanner model for waypoint prediction.")

    parser.add_argument("--exp_dir", type=str, default="logs", help="Directory to save logs and models.")
    parser.add_argument("--model_name", type=str, default="mlp_planner", choices=MODEL_FACTORY.keys(), help="Name of the model to train.")
    parser.add_argument("--num_epoch", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--transform_pipeline", type=str, default="default", choices=["default", "aug"], help="Data augmentation pipeline.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")

    args = parser.parse_args()

    train_planner(**vars(args))
