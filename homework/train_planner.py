# homework/train_planner.py

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .datasets.road_dataset import load_data
from .models import load_model, save_model
from .visualization import Visualizer
from .metrics import PlannerMetric
from .datasets.road_transforms import Normalize

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_mean_std(dataset_path: str, transform_pipeline: str = "state_only", batch_size: int = 32, num_workers: int = 4):
    """
    Computes the mean and standard deviation of track_left and track_right across the dataset.

    Args:
        dataset_path (str): Path to the dataset.
        transform_pipeline (str): Transform pipeline to use (should include track_left and track_right).
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: mean and std as numpy arrays of shape (2,) each.
    """
    dataloader = load_data(
        dataset_path=dataset_path,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    n_samples = 0
    sum_ = 0.0
    sum_sq = 0.0

    for batch in dataloader:
        # batch['track_left'] and batch['track_right'] have shape (B, 10, 2)
        track_left = batch['track_left']  # Tensor
        track_right = batch['track_right']  # Tensor

        # Concatenate along the second dimension to treat all points together
        tracks = torch.cat([track_left, track_right], dim=1)  # Shape: (B, 20, 2)
        tracks = tracks.view(-1, 2)  # Shape: (B*20, 2)

        n = tracks.size(0)
        sum_ += tracks.sum(dim=0)
        sum_sq += (tracks ** 2).sum(dim=0)
        n_samples += n

    mean = sum_ / n_samples
    std = torch.sqrt((sum_sq / n_samples) - (mean ** 2))

    mean_np = mean.numpy()
    std_np = std.numpy()

    print(f"Computed Mean: {mean_np}")
    print(f"Computed Std: {std_np}")

    # Save the statistics for later use
    stats_path = Path("homework/datasets/stats.npz")
    np.savez(stats_path, mean=mean_np, std=std_np)
    print(f"Saved statistics to {stats_path}")

    return mean_np, std_np


def plot_validation_samples(model, dataloader, device, epoch, plots_dir, num_samples=5):
    """
    Plots and saves a few samples from the validation set.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): Validation DataLoader.
        device (torch.device): Device to run the model on.
        epoch (int): Current epoch number.
        plots_dir (Path): Directory to save plots.
        num_samples (int): Number of samples to plot.
    """
    model.eval()
    viz = Visualizer()
    samples_plotted = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_plotted >= num_samples:
                break

            # Check if 'image' is in the batch
            if 'image' in batch:
                image = batch['image'].to(device)
            else:
                image = None

            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)

            preds = model(track_left, track_right)

            # Move tensors to CPU for visualization
            if image is not None:
                image_cpu = image.cpu()
            track_left_cpu = track_left.cpu()
            track_right_cpu = track_right.cpu()
            waypoints_cpu = waypoints.cpu()
            preds_cpu = preds.cpu()
            waypoints_mask_cpu = waypoints_mask.cpu()

            # Generate plot
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot left and right tracks
            ax.plot(track_left_cpu[0, :, 0], track_left_cpu[0, :, 1], 'ro-', label='Left Track')
            ax.plot(track_right_cpu[0, :, 0], track_right_cpu[0, :, 1], 'bo-', label='Right Track')

            # Plot ground truth waypoints
            ax.plot(waypoints_cpu[0, :, 0], waypoints_cpu[0, :, 1], 'g--o', label='Ground Truth Waypoints')

            # Plot predicted waypoints
            ax.plot(preds_cpu[0, :, 0], preds_cpu[0, :, 1], 'c--o', label='Predicted Waypoints')

            # Indicate masked waypoints
            for i, valid in enumerate(waypoints_mask_cpu[0]):
                if not valid:
                    ax.plot(waypoints_cpu[0, i, 0], waypoints_cpu[0, i, 1], 'kx', markersize=10, label='Masked Waypoints' if i == 0 else "")

            ax.set_title(f'Epoch {epoch+1} - Sample {samples_plotted+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid(True)

            # Save the plot
            plot_path = plots_dir / f'epoch_{epoch+1}_sample_{samples_plotted+1}.png'
            plt.savefig(plot_path)
            plt.close(fig)

            samples_plotted += 1


def main():
    # Set random seeds for reproducibility
    set_seed(42)

    parser = argparse.ArgumentParser(description="Train MLP Planner")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for TensorBoard logs.")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save validation plots.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training ('cuda' or 'cpu').")
    parser.add_argument("--save_every", type=int, default=10, help="Save model every N epochs.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping.")
    # Scheduler-specific arguments
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced. new_lr = lr * factor.")
    parser.add_argument("--lr_scheduler_patience", type=int, default=5, help="Number of epochs with no improvement after which learning rate will be reduced.")
    parser.add_argument("--lr_scheduler_threshold", type=float, default=1e-4, help="Threshold for measuring the new optimum, to only focus on significant changes.")
    parser.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6, help="A lower bound on the learning rate.")
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Create directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Step 1: Compute Mean and Std
    stats_path = Path("homework/datasets/stats.npz")
    if not stats_path.exists():
        print("Computing mean and std for normalization...")
        mean, std = compute_mean_std(
            dataset_path=args.train_dataset,
            transform_pipeline="state_only",
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        stats = np.load(stats_path)
        mean = stats["mean"]
        std = stats["std"]
        print(f"Loaded Mean: {mean}")
        print(f"Loaded Std: {std}")

    # Step 2: Load Data with Augmentation and Normalization for Training
    train_loader = load_data(
        dataset_path='drive_data/train',
        transform_pipeline="aug",
        return_dataloader=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        normalize=True,
        mean=mean,
        std=std,
    )

    # Load Validation Data without Augmentation
    val_loader = load_data(
        dataset_path='drive_data/val',
        transform_pipeline="state_only",
        return_dataloader=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # Step 3: Initialize Model
    model = load_model("mlp_planner").to(device)
    print(f"Model loaded and moved to {device}")

    # Step 4: Define Loss Function and Optimizer
    criterion = nn.L1Loss(reduction='none')  # We'll handle masking manually
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    print("Loss function and optimizer initialized.")

    # Step 5: Initialize Metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    # Step 6: Initialize Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # Assuming we monitor validation loss
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        threshold=args.lr_scheduler_threshold,
        min_lr=args.lr_scheduler_min_lr,
        verbose=True
    )
    print("Learning rate scheduler initialized.")

    # Step 7: Training Loop
    best_val_l1 = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = args.early_stopping_patience

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_metric.reset()

        for batch_idx, batch in enumerate(train_loader):
            track_left = batch['track_left'].to(device)  # Shape: (B, 10, 2)
            track_right = batch['track_right'].to(device)  # Shape: (B, 10, 2)
            waypoints = batch['waypoints'].to(device)  # Shape: (B, 3, 2)
            waypoints_mask = batch['waypoints_mask'].to(device)  # Shape: (B, 3)

            optimizer.zero_grad()

            preds = model(track_left, track_right)  # Shape: (B, 3, 2)

            # Compute loss with mask
            loss = criterion(preds, waypoints)  # Shape: (B, 3, 2)
            mask = waypoints_mask.unsqueeze(-1).float()  # Shape: (B, 3, 1)
            loss = (loss * mask).mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update training metrics
            train_metric.add(preds, waypoints, waypoints_mask)

        avg_train_loss = running_loss / len(train_loader)
        train_metrics = train_metric.compute()

        print(f"Epoch [{epoch+1}/{args.epochs}], Training Loss: {avg_train_loss:.4f}, L1 Error: {train_metrics['l1_error']:.4f}")

        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch+1)
        writer.add_scalar('Metrics/L1_Error_Train', train_metrics['l1_error'], epoch+1)
        writer.add_scalar('Metrics/Longitudinal_Error_Train', train_metrics['longitudinal_error'], epoch+1)
        writer.add_scalar('Metrics/Lateral_Error_Train', train_metrics['lateral_error'], epoch+1)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_metric.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)

                preds = model(track_left, track_right)

                # Compute loss with mask
                loss = criterion(preds, waypoints)
                mask = waypoints_mask.unsqueeze(-1).float()
                loss = (loss * mask).mean()

                val_running_loss += loss.item()

                # Update validation metrics
                val_metric.add(preds, waypoints, waypoints_mask)

        avg_val_loss = val_running_loss / len(val_loader)
        val_metrics = val_metric.compute()

        print(f"Epoch [{epoch+1}/{args.epochs}], Validation Loss: {avg_val_loss:.4f}, L1 Error: {val_metrics['l1_error']:.4f}")

        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch+1)
        writer.add_scalar('Metrics/L1_Error_Validation', val_metrics['l1_error'], epoch+1)
        writer.add_scalar('Metrics/Longitudinal_Error_Validation', val_metrics['longitudinal_error'], epoch+1)
        writer.add_scalar('Metrics/Lateral_Error_Validation', val_metrics['lateral_error'], epoch+1)

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Log the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch+1)
        print(f"Current Learning Rate: {current_lr}")

        # Save validation plots
        plot_validation_samples(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            plots_dir=Path(args.plots_dir),
            num_samples=5
        )

        # Check for improvement
        if val_metrics['l1_error'] < best_val_l1:
            best_val_l1 = val_metrics['l1_error']
            save_model(model)
            print(f"Best model saved at epoch {epoch+1} with L1 Error: {best_val_l1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in L1 Error for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Optionally, save model every N epochs
        if (epoch + 1) % args.save_every == 0:
            save_path = Path(args.plots_dir) / f'mlp_planner_epoch_{epoch+1}.th'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
