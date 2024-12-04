from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


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
        n_track: int = 10,          # Number of track points on each side
        n_waypoints: int = 3,       # Number of waypoints to predict
        d_model: int = 64,          # Dimension of latent embeddings
        num_heads: int = 16,         # Number of attention heads
        num_blocks: int = 6,        # Number of Perceiver blocks
        dim_feedforward: int = 256, # Dimension of feedforward network in SelfAttention
        dropout: float = 0.1,       # Dropout rate
    ):
        """
        Initializes the PerceiverPlanner.
    
        Args:
            n_track (int): Number of track points on each side (left and right).
            n_waypoints (int): Number of waypoints to predict.
            d_model (int): Dimension of latent embeddings.
            num_heads (int): Number of attention heads.
            num_blocks (int): Number of Perceiver blocks.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """
        super(TransformerPlanner, self).__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        input_dim = 2  # Each lane point has 2D coordinates

        # Initialize Perceiver model
        self.perceiver = Perceiver(
            input_dim=input_dim,
            latent_dim=d_model,
            num_latents=n_waypoints,
            num_blocks=num_blocks,
            num_heads=num_heads,
            output_dim=2,  # Output is 2D waypoints
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.
    
        Args:
            track_left (torch.Tensor): shape (batch_size, n_track, 2)
            track_right (torch.Tensor): shape (batch_size, n_track, 2)
    
        Returns:
            torch.Tensor: future waypoints with shape (batch_size, n_waypoints, 2)
        """
        batch_size = track_left.size(0)

        # 1. Concatenate Left and Right Tracks
        # Shape: (batch_size, 2 * n_track, 2)
        track = torch.cat([track_left, track_right], dim=1)
        # print(f"{torch.std_mean(track) = }")
        # 2. Pass through Perceiver
        # Output shape: (batch_size, n_waypoints, 2)
        waypoints = self.perceiver(track)
        # print(f"{torch.std_mean(waypoints) = }")
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


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

# ======================================================================
#                           Perciever Components
# ======================================================================

# homework/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, input_dim, num_heads, dropout=0.1):
        """
        Cross-Attention module where latents attend to input features with LayerNorm.
    
        Args:
            latent_dim (int): Dimension of latent embeddings.
            input_dim (int): Dimension of input features.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(CrossAttention, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.to_latent = nn.Linear(input_dim, latent_dim)
        
        # LayerNorm for Cross-Attention
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, latents, inputs, input_mask=None):
        """
        Forward pass for Cross-Attention with residual connection and LayerNorm.
    
        Args:
            latents (torch.Tensor): Latent embeddings, shape (batch_size, num_latents, latent_dim)
            inputs (torch.Tensor): Input features, shape (batch_size, input_seq_len, input_dim)
            input_mask (torch.Tensor, optional): Mask for input features, shape (batch_size, input_seq_len)
    
        Returns:
            torch.Tensor: Updated latents after cross-attention, shape (batch_size, num_latents, latent_dim)
        """
        # Project inputs to latent dimension
        inputs_proj = self.to_latent(inputs)  # Shape: (batch_size, input_seq_len, latent_dim)
    
        # Perform cross-attention: queries=latents, keys=values=inputs_proj
        attn_output, attn_weights = self.attention(query=latents, key=inputs_proj, value=inputs_proj, key_padding_mask=input_mask)
    
        # Apply Dropout
        attn_output = self.dropout_layer(attn_output)
    
        # Residual Connection and LayerNorm
        latents = self.norm(latents + attn_output)  # Shape: (batch_size, num_latents, latent_dim)
    
        return latents
    

class SelfAttention(nn.Module):
    def __init__(self, latent_dim, num_heads, dropout=0.1):
        """
        Self-Attention module for processing latents.

        Args:
            latent_dim (int): Dimension of latent embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(SelfAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.linear1 = nn.Linear(latent_dim, latent_dim * 4)
        self.linear2 = nn.Linear(latent_dim * 4, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.activation = F.relu

    def forward(self, x):
        """
        Forward pass for Self-Attention.

        Args:
            x (torch.Tensor): Latent embeddings, shape (batch_size, num_latents, latent_dim)

        Returns:
            torch.Tensor: Updated latents after self-attention, shape (batch_size, num_latents, latent_dim)
        """
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + ff_output)

        return x


class PerceiverBlock(nn.Module):
    def __init__(self, latent_dim, input_dim, num_heads, dropout=0.1):
        """
        A single Perceiver block consisting of cross-attention and self-attention.
    
        Args:
            latent_dim (int): Dimension of latent embeddings.
            input_dim (int): Dimension of input features.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(PerceiverBlock, self).__init__()
        self.cross_attn = CrossAttention(latent_dim, input_dim, num_heads, dropout)
        self.self_attn = SelfAttention(latent_dim, num_heads, dropout)

    def forward(self, latents, inputs, input_mask=None):
        """
        Forward pass for PerceiverBlock.
    
        Args:
            latents (torch.Tensor): Latent embeddings, shape (batch_size, num_latents, latent_dim)
            inputs (torch.Tensor): Input features, shape (batch_size, input_seq_len, input_dim)
            input_mask (torch.Tensor, optional): Mask for input features, shape (batch_size, input_seq_len)
    
        Returns:
            torch.Tensor: Updated latents after PerceiverBlock, shape (batch_size, num_latents, latent_dim)
        """
        latents = self.cross_attn(latents, inputs, input_mask)  # LayerNorm applied within CrossAttention
        latents = self.self_attn(latents)  # LayerNorm applied within SelfAttention
        return latents


class Perceiver(nn.Module):
    def __init__(self, 
                 input_dim,        # Dimension of input features
                 latent_dim,       # Dimension of latent array
                 num_latents,      # Number of latent vectors (e.g., number of waypoints)
                 num_blocks,       # Number of Perceiver blocks
                 num_heads,        # Number of attention heads
                 output_dim,       # Dimension of output features (e.g., 2 for 2D waypoints)
                 ):
        """
        Perceiver model for processing input features and predicting outputs.

        Args:
            input_dim (int): Dimension of input features (e.g., 2 for 2D lane points).
            latent_dim (int): Dimension of latent embeddings.
            num_latents (int): Number of latent vectors (e.g., number of waypoints to predict).
            num_blocks (int): Number of Perceiver blocks.
            num_heads (int): Number of attention heads.
            output_dim (int): Dimension of output features (e.g., 2 for 2D waypoints).
        """
        super(Perceiver, self).__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))  # Initialized latents
        self.blocks = nn.ModuleList([
            PerceiverBlock(latent_dim, input_dim, num_heads) for _ in range(num_blocks)
        ])
        self.to_output = nn.Linear(latent_dim, output_dim)  # Project latents to output features

    def forward(self, inputs, input_mask=None):
        """
        Forward pass for Perceiver.

        Args:
            inputs (torch.Tensor): Input features, shape (batch_size, input_seq_len, input_dim)
            input_mask (torch.Tensor, optional): Mask for input features, shape (batch_size, input_seq_len)

        Returns:
            torch.Tensor: Predicted outputs, shape (batch_size, num_latents, output_dim)
        """
        batch_size = inputs.size(0)
        latents = self.latents.expand(batch_size, -1, -1)  # Shape: (batch_size, num_latents, latent_dim)

        for block in self.blocks:
            latents = block(latents, inputs, input_mask)

        # Project latents to output features
        output = self.to_output(latents)  # Shape: (batch_size, num_latents, output_dim)
        return output
