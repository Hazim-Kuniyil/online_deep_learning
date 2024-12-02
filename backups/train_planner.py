# homework/train_planner.py

import argparse
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

from homework.models import load_model, save_model, MODEL_FACTORY
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt

from torch.cuda.amp import GradScaler, autocast  # For mixed precision


def visualize_predictions(model, data_loader, device, num_samples=5):
    model.eval()
    samples_visualized = 0
    figs_dir = Path('figs')
    figs_dir.mkdir(parents=True, exist_ok=True)  # Ensure 'figs' directory exists
    with torch.no_grad():
        for batch in data_loader:
            track_left = batch['track_left'].to(device)          # Shape: (B, n_track, 2)
            track_right = batch['track_right'].to(device)        # Shape: (B, n_track, 2)
            waypoints_gt = batch['waypoints'].to(device)         # Shape: (B, n_waypoints, 2)
            waypoints_mask = batch['waypoints_mask'].to(device)  # Shape: (B, n_waypoints)
            waypoints_pred = model(track_left, track_right)      # Shape: (B, n_waypoints, 2)
            
            for i in range(track_left.size(0)):
                plt.figure(figsize=(8, 6))
                
                # Plot Left Track
                plt.plot(
                    track_left[i, :, 0].cpu(),
                    track_left[i, :, 1].cpu(),
                    color='blue',
                    linestyle='-',
                    linewidth=2,
                    label='Left Track'
                )
                
                # Plot Right Track
                plt.plot(
                    track_right[i, :, 0].cpu(),
                    track_right[i, :, 1].cpu(),
                    color='orange',
                    linestyle='-',
                    linewidth=2,
                    label='Right Track'
                )
                
                # Plot Ground Truth Waypoints
                plt.scatter(
                    waypoints_gt[i, :, 0].cpu(),
                    waypoints_gt[i, :, 1].cpu(),
                    c='green',
                    marker='o',
                    label='Ground Truth Waypoints'
                )
                
                # Plot Predicted Waypoints
                plt.scatter(
                    waypoints_pred[i, :, 0].cpu(),
                    waypoints_pred[i, :, 1].cpu(),
                    c='red',
                    marker='x',
                    label='Predicted Waypoints'
                )
                
                # Optionally, connect waypoints for Ground Truth
                plt.plot(
                    waypoints_gt[i, :, 0].cpu(),
                    waypoints_gt[i, :, 1].cpu(),
                    color='green',
                    linestyle='--',
                    linewidth=1
                )
                
                # Optionally, connect waypoints for Predictions
                plt.plot(
                    waypoints_pred[i, :, 0].cpu(),
                    waypoints_pred[i, :, 1].cpu(),
                    color='red',
                    linestyle='--',
                    linewidth=1
                )
                
                plt.title('Waypoint Prediction vs Ground Truth with Track Boundaries')
                plt.xlabel('X-coordinate')
                plt.ylabel('Y-coordinate')
                plt.legend()
                plt.grid(True)
                
                # Create a unique filename by combining timestamp and sample index
                timestamp = datetime.now().strftime('%m%d_%H%M%S')
                plt.savefig(figs_dir / f"mlp_{timestamp}_sample_{i}.png")
                plt.close()
                
                samples_visualized += 1
                if samples_visualized >= num_samples:
                    return


# Removed the custom PerWaypointL1Loss
# class PerWaypointL1Loss(nn.Module):
#     ...


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train_planner(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
    transform_pipeline: str = "aug",
    num_workers: int = 4,
    scheduler_step_size: int = 30,  # Learning rate scheduler step size
    scheduler_gamma: float = 0.1,    # Learning rate scheduler decay factor
    max_grad_norm: float = 1.0,      # Gradient clipping max norm
    dropout_rate: float = 0.3,        # Reduced Dropout probability
    early_stopping_patience: int = 10, # Early stopping patience
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
        scheduler_step_size (int): Period of learning rate decay (in epochs).
        scheduler_gamma (float): Multiplicative factor of learning rate decay.
        max_grad_norm (float): Maximum norm for gradient clipping.
        dropout_rate (float): Dropout probability for hidden layers.
        early_stopping_patience (int): Patience for early stopping.
        **kwargs: Additional keyword arguments for model initialization.
    """
    # ========================
    # 1. Setup and Initialization
    # ========================

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

    # Initialize the model with Dropout
    model = load_model(model_name, dropout_rate=dropout_rate, **kwargs)
    model = model.to(device)
    print(f"Initialized model: {model_name} with dropout_rate={dropout_rate}")

    # Apply weight initialization
    model.apply(initialize_weights)
    print("Applied Kaiming Normal initialization to model weights.")

    # ========================
    # 2. Define Loss Function and Optimizer
    # ========================

    # Define the standard L1 loss function
    criterion = nn.L1Loss(reduction='mean')
    print("Defined standard nn.L1Loss with 'mean' reduction.")

    # Define the optimizer (AdamW) with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    print("Defined optimizer: AdamW with weight decay.")

    # ========================
    # 3. Define Learning Rate Scheduler
    # ========================

    # Initialize the Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    print(f"Initialized StepLR scheduler with step_size={scheduler_step_size} and gamma={scheduler_gamma}.")

    # ========================
    # 4. Load Data
    # ========================

    # Load training and validation data
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,  # Use 'aug' for training with augmentation
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline="default",  # Use 'default' for validation without augmentation
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print("Loaded training and validation data.")

    # ========================
    # 5. Initialize Metrics
    # ========================

    # Initialize metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    # ========================
    # 6. Initialize Early Stopping
    # ========================

    best_val_l1_error = float('inf')  # Lower is better
    no_improve_epochs = 0
    print(f"Early stopping patience set to {early_stopping_patience} epochs.")

    # ========================
    # 7. Initialize GradScaler for Mixed Precision
    # ========================

    scaler = GradScaler()

    # ========================
    # 8. Training Loop
    # ========================

    global_step = 0

    for epoch in range(1, num_epoch + 1):
        print(f"\nEpoch {epoch}/{num_epoch}")
        model.train()
        train_metric.reset()

        # Training Loop
        for batch_idx, batch in enumerate(train_loader):
            track_left = batch['track_left'].to(device)          # Shape: (B, n_track, 2)
            track_right = batch['track_right'].to(device)        # Shape: (B, n_track, 2)
            waypoints_gt = batch['waypoints'].to(device)         # Shape: (B, n_waypoints, 2)
            waypoints_mask = batch['waypoints_mask'].to(device)  # Shape: (B, n_waypoints)

            optimizer.zero_grad()

            with autocast():
                # Forward pass
                waypoints_pred = model(track_left, track_right)      # Shape: (B, n_waypoints, 2)
                # Compute standard L1 loss
                loss = nn.L1Loss(reduction='none')(waypoints_pred, waypoints_gt)  # Shape: (B, N, D)
                if waypoints_mask is not None:
                    loss = loss * waypoints_mask.unsqueeze(2)  # Shape: (B, N, D)
                    loss = loss.sum(dim=2)  # Shape: (B, N)
                    loss = loss.sum() / waypoints_mask.sum()
                else:
                    loss = loss.mean()

            # Backward pass and optimization with mixed precision
            scaler.scale(loss).backward()

            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            # Update metrics
            train_metric.add(preds=waypoints_pred, labels=waypoints_gt, labels_mask=waypoints_mask)

            # Compute gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            logger.add_scalar('Train/Gradient_Norm', total_norm, global_step)

            # Logging
            logger.add_scalar('Train/Loss', loss.item(), global_step)

            if global_step % 100 == 0:
                current_metrics = train_metric.compute()
                print(f"Step {global_step}: Loss={loss.item():.4f}, L1_Error={current_metrics['l1_error']:.4f}")
                logger.add_scalar('Train/L1_Error', current_metrics['l1_error'], global_step)
                logger.flush()

            global_step += 1

        # ========================
        # 9. Post-Epoch Operations
        # ========================

        # Compute training metrics after epoch
        train_metrics = train_metric.compute()
        logger.add_scalar('Epoch/Train_L1_Error', train_metrics['l1_error'], epoch)
        logger.add_scalar('Epoch/Train_Longitudinal_Error', train_metrics['longitudinal_error'], epoch)
        logger.add_scalar('Epoch/Train_Lateral_Error', train_metrics['lateral_error'], epoch)
        print(f"Training Metrics: L1_Error={train_metrics['l1_error']:.4f}, "
              f"Longitudinal_Error={train_metrics['longitudinal_error']:.4f}, "
              f"Lateral_Error={train_metrics['lateral_error']:.4f}")

        # Validation Loop
        model.eval()
        val_metric.reset()
        with torch.no_grad():
            # Visualize predictions on validation set
            visualize_predictions(model, val_loader, device)

            for batch in val_loader:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints_gt = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)

                waypoints_pred = model(track_left, track_right)

                # Compute standard L1 loss
                loss = nn.L1Loss(reduction='none')(waypoints_pred, waypoints_gt)  # Shape: (B, N, D)
                if waypoints_mask is not None:
                    loss = loss * waypoints_mask.unsqueeze(2)  # Shape: (B, N, D)
                    loss = loss.sum(dim=2)  # Shape: (B, N)
                    loss = loss.sum() / waypoints_mask.sum()
                else:
                    loss = loss.mean()
                logger.add_scalar('Val/Loss', loss.item(), epoch)

                # Update metrics
                val_metric.add(preds=waypoints_pred, labels=waypoints_gt, labels_mask=waypoints_mask)

        # Compute validation metrics after epoch
        val_metrics = val_metric.compute()
        logger.add_scalar('Epoch/Val_L1_Error', val_metrics['l1_error'], epoch)
        logger.add_scalar('Epoch/Val_Longitudinal_Error', val_metrics['longitudinal_error'], epoch)
        logger.add_scalar('Epoch/Val_Lateral_Error', val_metrics['lateral_error'], epoch)
        print(f"Validation Metrics: L1_Error={val_metrics['l1_error']:.4f}, "
              f"Longitudinal_Error={val_metrics['longitudinal_error']:.4f}, "
              f"Lateral_Error={val_metrics['lateral_error']:.4f}")

        # ========================
        # 10. Early Stopping Check
        # ========================

        # Save the best model based on validation L1 error
        if val_metrics['l1_error'] < best_val_l1_error:
            best_val_l1_error = val_metrics['l1_error']
            no_improve_epochs = 0
            best_model_path = log_dir / f"best_model_epoch_{epoch}.th"
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch} with L1_Error={best_val_l1_error:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement in validation L1 error for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= early_stopping_patience:
                print(f"No improvement in validation L1 error for {early_stopping_patience} epochs. Stopping training.")
                break

        # ========================
        # 11. Model Checkpointing
        # ========================

        # Save checkpoints every 10 epochs and the final epoch
        if epoch % 10 == 0 or epoch == num_epoch:
            checkpoint_path = log_dir / f"epoch_{epoch}.th"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # ========================
        # 12. Step the Learning Rate Scheduler
        # ========================

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
        print(f"Updated Learning Rate to: {current_lr}")

    # ========================
    # 13. Final Model Saving and Cleanup
    # ========================

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
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--transform_pipeline", type=str, default="aug", choices=["default", "aug"], help="Data augmentation pipeline.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--scheduler_step_size", type=int, default=30, help="Period of learning rate decay (in epochs).")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1, help="Multiplicative factor of learning rate decay.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum norm for gradient clipping.")
    parser.add_argument("--dropout_rate", type=float, default=0, help="Dropout probability for hidden layers.")  # Reduced Dropout rate
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping.")

    args = parser.parse_args()

    train_planner(**vars(args))
