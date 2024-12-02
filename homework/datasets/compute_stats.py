# homework/datasets/compute_stats.py

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from road_dataset import RoadDataset, load_data
from road_transforms import EgoTrackProcessor

def compute_mean_std(dataset_path: str, transform_pipeline: str = "state_only", batch_size: int = 32, num_workers: int = 2):
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
    # Load dataset with state_only pipeline
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

        # Concatenate along the first dimension to treat all points together
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
    stats_path = Path(__file__).resolve().parent / "stats.npz"
    np.savez(stats_path, mean=mean_np, std=std_np)
    print(f"Saved statistics to {stats_path}")

    return mean_np, std_np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute dataset mean and std for normalization.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--transform_pipeline", type=str, default="state_only", help="Transform pipeline to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader.")

    args = parser.parse_args()

    compute_mean_std(
        dataset_path=args.dataset_path,
        transform_pipeline=args.transform_pipeline,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
