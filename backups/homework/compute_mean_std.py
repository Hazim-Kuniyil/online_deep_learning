# homework/compute_mean_std.py

import torch
from homework.datasets.road_dataset import load_data

def compute_mean_std(train_loader):
    sum_ = 0.0
    sum_sq = 0.0
    n = 0
    for batch in train_loader:
        track_left = batch['track_left']  # Shape: (B, 10, 2)
        track_right = batch['track_right']  # Shape: (B, 10, 2)
        
        # Concatenate track_left and track_right for overall statistics
        tracks = torch.cat([track_left, track_right], dim=1)  # Shape: (B, 20, 2)
        
        sum_ += tracks.sum(dim=(0, 1))
        sum_sq += (tracks ** 2).sum(dim=(0, 1))
        n += tracks.numel()
    
    mean = sum_ / n
    std = torch.sqrt((sum_sq / n) - (mean ** 2))
    return mean.tolist(), std.tolist()

def main():
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline="state_only",  # No augmentations for accurate stats
        shuffle=False,
        batch_size=32,
        num_workers=4,
    )
    mean, std = compute_mean_std(train_loader)
    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")

if __name__ == "__main__":
    main()
