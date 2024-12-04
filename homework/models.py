from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


# models.py

import torch
import torch.nn as nn

class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 512,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): number of hidden units in the MLP
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 4  # (track_left + track_right) each with 2 features
        output_dim = n_waypoints * 2  # each waypoint has 2 coordinates

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        waypoints_mask: torch.Tensor = None,  # Optional: for masking during loss computation
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (B, n_track, 2)
            track_right (torch.Tensor): shape (B, n_track, 2)
            waypoints_mask (torch.Tensor, optional): shape (n_waypoints,), mask indicating "clean" waypoints

        Returns:
            torch.Tensor: future waypoints with shape (B, n_waypoints, 2)
        """
        # Concatenate track boundaries along the last dimension
        # Resulting shape: (B, n_track, 4)
        x = torch.cat([track_left, track_right], dim=-1)

        # Flatten the tensor to shape (B, n_track * 4)
        x = x.view(x.size(0), -1)

        # Pass through the MLP
        x = self.mlp(x)  # Shape: (B, n_waypoints * 2)

        # Reshape to (B, n_waypoints, 2)
        x = x.view(x.size(0), self.n_waypoints, 2)

        return x


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_track (int): Number of points in each side of the track.
            n_waypoints (int): Number of waypoints to predict.
            d_model (int): Dimension of the model embeddings.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer decoder layers.
            dim_feedforward (int): Dimension of the feedforward network in the transformer.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Input projection for lane boundary points
        self.input_proj = nn.Linear(2, d_model)

        # Learnable query embeddings for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to 2D coordinates
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): shape (B, n_track, 2)
            track_right (torch.Tensor): shape (B, n_track, 2)

        Returns:
            torch.Tensor: Future waypoints with shape (B, n_waypoints, 2)
        """
        B = track_left.size(0)

        # Concatenate left and right tracks: (B, n_track*2, 2)
        track_combined = torch.cat([track_left, track_right], dim=1)  # (B, 20, 2)

        # Project input points to d_model
        # (B, n_track*2, d_model)
        track_embedded = self.input_proj(track_combined)

        # Prepare memory for transformer: shape (n_track*2, B, d_model)
        memory = track_embedded.permute(1, 0, 2)  # (S, B, E)

        # Prepare query embeddings
        # (n_waypoints, B, d_model)
        query_embeddings = self.query_embed.weight  # (n_waypoints, d_model)
        query_embeddings = query_embeddings.unsqueeze(1).repeat(1, B, 1)  # (n_waypoints, B, d_model)

        # Transformer expects tgt to be (T, B, E)
        decoded = self.transformer_decoder(tgt=query_embeddings, memory=memory)  # (n_waypoints, B, d_model)

        # Permute to (B, n_waypoints, d_model)
        decoded = decoded.permute(1, 0, 2)  # (B, n_waypoints, d_model)

        # Project to 2D coordinates
        waypoints = self.output_proj(decoded)  # (B, n_waypoints, 2)

        return waypoints


class CNNPlanner(nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Define CNN architecture
        # Example architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC -> ReLU -> FC
        # Final output layer has n_waypoints * 2 neurons

        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2),  # (B, 16, 48, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 16, 24, 32)

            # Second convolutional block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # (B, 32, 12, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 32, 6, 8)

            # Third convolutional block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # (B, 64, 3, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 64, 1, 1)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # (B, 64)
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2),  # Output: (B, n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predicts waypoints from image input.

        Args:
            image (torch.FloatTensor): shape (B, 3, 96, 128) and values in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (B, n_waypoints, 2)
        """
        # Normalize the image
        # Shape: (B, 3, 96, 128)
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through CNN
        x = self.conv_layers(x)  # Shape: (B, 64, 1, 1)

        # Pass through fully connected layers
        x = self.fc_layers(x)  # Shape: (B, n_waypoints * 2)

        # Reshape to (B, n_waypoints, 2)
        waypoints = x.view(-1, self.n_waypoints, 2)

        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
