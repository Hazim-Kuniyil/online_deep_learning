# homework/train_planner.py

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .metrics import PlannerMetric
from .models import MODEL_FACTORY, save_model, load_model
from .datasets.road_dataset import load_data

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Planner Models")
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mlp_planner", "transformer_planner", "cnn_planner"],
        help="Model to train: 'mlp_planner', 'transformer_planner', or 'cnn_planner'",
    )
    
    # Data paths
    parser.add_argument(
        "--train_data",
        type=str,
        default="drive_data/train",
        help="Path to training data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="drive_data/val",
        help="Path to validation data",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization) for optimizer",
    )
    
    # Transformer-specific hyperparameters
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Dimension of model embeddings (Transformer only)",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of attention heads (Transformer only)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of transformer decoder layers (Transformer only)",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=256,
        help="Dimension of feedforward network in Transformer (Transformer only)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (Transformer only)",
    )
    
    # Other options
    parser.add_argument(
        "--save_dir",
        type=str,
        default="trained_models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name: str, args):
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Model '{model_name}' not found in MODEL_FACTORY.")
    
    if model_name == "mlp_planner":
        model = MODEL_FACTORY[model_name](
            n_track=10,
            n_waypoints=3,
            hidden_sizes=[128, 64],  # You can adjust hidden layer sizes
        )
    elif model_name == "transformer_planner":
        model = MODEL_FACTORY[model_name](
            n_track=10,
            n_waypoints=3,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )
    elif model_name == "cnn_planner":
        model = MODEL_FACTORY[model_name](
            n_waypoints=3,
        )
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    
    return model


def train_one_epoch(model, dataloader, optimizer, criterion, metric, device, verbose=False):
    model.train()
    running_loss = 0.0
    metric.reset()
    
    for batch_idx, batch in enumerate(dataloader):
        if model.__class__.__name__ == "CNNPlanner":
            # For CNNPlanner, use image as input
            image = batch['image'].to(device)              # Shape: (B, 3, 96, 128)
            waypoints = batch['waypoints'].to(device)      # Shape: (B, 3, 2)
            mask = batch['waypoints_mask'].to(device)      # Shape: (B, 3)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(image)                            # Shape: (B, 3, 2)
        
        else:
            # For MLPPlanner and TransformerPlanner
            track_left = batch['track_left'].to(device)      # Shape: (B, 10, 2)
            track_right = batch['track_right'].to(device)    # Shape: (B, 10, 2)
            waypoints = batch['waypoints'].to(device)        # Shape: (B, 3, 2)
            mask = batch['waypoints_mask'].to(device)        # Shape: (B, 3)
        
            optimizer.zero_grad()
        
            # Forward pass
            preds = model(track_left, track_right)           # Shape: (B, 3, 2)
        
        # Compute loss (only on masked waypoints)
        loss = criterion(preds, waypoints)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * waypoints.size(0)
        
        # Update metrics
        metric.add(preds, waypoints, mask)
        
        if verbose and (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = metric.compute()
    
    return epoch_loss, metrics


def evaluate(model, dataloader, criterion, metric, device, verbose=False):
    model.eval()
    running_loss = 0.0
    metric.reset()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if model.__class__.__name__ == "CNNPlanner":
                # For CNNPlanner, use image as input
                image = batch['image'].to(device)              # Shape: (B, 3, 96, 128)
                waypoints = batch['waypoints'].to(device)      # Shape: (B, 3, 2)
                mask = batch['waypoints_mask'].to(device)      # Shape: (B, 3)
                
                # Forward pass
                preds = model(image)                            # Shape: (B, 3, 2)
        
            else:
                # For MLPPlanner and TransformerPlanner
                track_left = batch['track_left'].to(device)      # Shape: (B, 10, 2)
                track_right = batch['track_right'].to(device)    # Shape: (B, 10, 2)
                waypoints = batch['waypoints'].to(device)        # Shape: (B, 3, 2)
                mask = batch['waypoints_mask'].to(device)        # Shape: (B, 3)
        
                # Forward pass
                preds = model(track_left, track_right)           # Shape: (B, 3, 2)
        
            # Compute loss
            loss = criterion(preds, waypoints)
            running_loss += loss.item() * waypoints.size(0)
        
            # Update metrics
            metric.add(preds, waypoints, mask)
        
            if verbose and (batch_idx + 1) % 100 == 0:
                print(f"  Eval Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = metric.compute()
    
    return epoch_loss, metrics


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    if args.verbose:
        print("Loading training data...")
    train_loader = load_data(
        dataset_path=args.train_data,
        transform_pipeline="default" if args.model == "cnn_planner" else "state_only",  # 'default' for CNNPlanner
        return_dataloader=True,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    if args.verbose:
        print("Loading validation data...")
    val_loader = load_data(
        dataset_path=args.val_data,
        transform_pipeline="default" if args.model == "cnn_planner" else "state_only",
        return_dataloader=True,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Initialize model
    if args.verbose:
        print(f"Initializing model '{args.model}'...")
    model = get_model(args.model, args)
    model = model.to(device)
    
    if args.verbose:
        print(f"Model architecture:\n{model}")
    
    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()
    
    best_val_loss = float('inf')
    best_epoch = -1
    
    # Training loop
    for epoch in range(1, args.num_epochs + 1):
        if args.verbose:
            print(f"\nEpoch {epoch}/{args.num_epochs}")
            print("-" * 20)
        
        # Training phase
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, train_metric, device, verbose=args.verbose
        )
        
        # Validation phase
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, val_metric, device, verbose=args.verbose
        )
        
        # Logging
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train L1 Error: {train_metrics['l1_error']:.4f} | "
              f"Longitudinal: {train_metrics['longitudinal_error']:.4f} | "
              f"Lateral: {train_metrics['lateral_error']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val L1 Error: {val_metrics['l1_error']:.4f} | "
              f"Longitudinal: {val_metrics['longitudinal_error']:.4f} | "
              f"Lateral: {val_metrics['lateral_error']:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save the best model
            save_path = save_model(model)
            # Move the saved model to the specified save directory with a unique name
            # model_filename = f"{args.model}_best_epoch_{epoch}.th"
            # final_save_path = Path(args.save_dir) / model_filename
            # Path(save_path).rename(final_save_path)
            if args.verbose:
                print(f"  Saved best model to {save_path}")
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}.")
    print(f"Best model saved to {Path(args.save_dir) / f'{args.model}_best_epoch_{best_epoch}.th'}")


if __name__ == "__main__":
    main()
