"""
PointNet Encoder + SIREN Decoder for tactile SDF reconstruction.

Architecture:
    PointNet: per-point MLP + max-pool → 256D global latent code
    SIREN:    4 hidden layers with sin() activations, conditioned on latent code
              maps (query_xyz, latent) → SDF value

References:
    - Qi et al., "PointNet" (CVPR 2017)
    - Sitzmann et al., "Implicit Neural Representations with Periodic
      Activation Functions" (NeurIPS 2020)
"""
import math
import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# SIREN building blocks
# ---------------------------------------------------------------------------

class SineActivation(nn.Module):
    """Sine activation function with configurable frequency."""

    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


class SirenLayer(nn.Module):
    """
    Linear layer with SIREN-style initialization and sine activation.

    For the first layer, weights are drawn from U(-1/in, 1/in).
    For subsequent layers, weights are drawn from
        U(-sqrt(6/in)/omega_0, sqrt(6/in)/omega_0).
    """

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False,
                 bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.omega_0 = omega_0
        self.is_first = is_first
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            dim = self.linear.in_features
            if self.is_first:
                bound = 1.0 / dim
            else:
                bound = math.sqrt(6.0 / dim) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# ---------------------------------------------------------------------------
# PointNet Encoder
# ---------------------------------------------------------------------------

class PointNetEncoder(nn.Module):
    """
    Simplified PointNet encoder for sparse contact points.

    Input:  (B, N_contacts, D_features)  e.g. (B, 6, 9)
    Output: (B, latent_dim) global feature vector
    """

    def __init__(self, input_dim: int = 9, latent_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),

            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),

            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),

            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, contacts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            contacts: (B, N, D) contact features
        Returns:
            (B, latent_dim) global feature via symmetric max-pooling
        """
        # Per-point features
        per_point = self.mlp(contacts)        # (B, N, latent_dim)
        # Symmetric aggregation
        global_feat, _ = per_point.max(dim=1)  # (B, latent_dim)
        return global_feat


# ---------------------------------------------------------------------------
# SIREN Decoder
# ---------------------------------------------------------------------------

class SirenDecoder(nn.Module):
    """
    SIREN-based decoder mapping (query_xyz, latent) → SDF.

    Input:  query coordinates (B, N, 3), latent code (B, latent_dim)
    Output: predicted SDF values (B, N, 1)
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256,
                 n_layers: int = 4, omega_0: float = 30.0):
        super().__init__()

        self.first_layer = SirenLayer(
            3 + latent_dim, hidden_dim, omega_0=omega_0, is_first=True
        )

        hidden_layers = []
        for _ in range(n_layers - 1):
            hidden_layers.append(
                SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0, is_first=False)
            )
        self.hidden = nn.Sequential(*hidden_layers)

        # Final linear layer (no sine activation) → SDF value
        self.output_layer = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.zero_()

    def forward(self, query_points: torch.Tensor,
                latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_points: (B, N, 3) 3D coordinates
            latent:       (B, latent_dim) global latent code
        Returns:
            (B, N, 1) predicted SDF values
        """
        B, N, _ = query_points.shape

        # Expand latent to match query points: (B, N, latent_dim)
        z = latent.unsqueeze(1).expand(-1, N, -1)

        # Concatenate: (B, N, 3 + latent_dim)
        x = torch.cat([query_points, z], dim=-1)

        # Forward through SIREN
        x = self.first_layer(x)
        x = self.hidden(x)
        sdf = self.output_layer(x)  # (B, N, 1)

        return sdf


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class TactileSDF(nn.Module):
    """
    Complete PointNet + SIREN model for tactile SDF reconstruction.

    Input:
        - contacts: (B, 6, 9) — 6 contact points with pos+normal+tangent
        - query_points: (B, N, 3) — 3D query coordinates

    Output:
        - sdf_pred: (B, N, 1) — predicted SDF at each query point
        - latent: (B, 256) — global latent code (for analysis)
    """

    def __init__(self, contact_dim: int = 9, latent_dim: int = 256,
                 hidden_dim: int = 256, n_siren_layers: int = 4,
                 omega_0: float = 30.0):
        super().__init__()
        self.encoder = PointNetEncoder(input_dim=contact_dim,
                                        latent_dim=latent_dim)
        self.decoder = SirenDecoder(latent_dim=latent_dim,
                                     hidden_dim=hidden_dim,
                                     n_layers=n_siren_layers,
                                     omega_0=omega_0)

    def forward(self, contacts: torch.Tensor,
                query_points: torch.Tensor):
        latent = self.encoder(contacts)
        sdf_pred = self.decoder(query_points, latent)
        return sdf_pred, latent

    def predict_grid(self, contacts: torch.Tensor,
                     resolution: int = 64,
                     bounds: float = 1.1,
                     device: str = "cpu") -> np.ndarray:
        """
        Predict SDF on a regular 3D grid for marching cubes.

        Args:
            contacts: (1, 6, 9) single sample contacts
            resolution: grid resolution per axis
            bounds: half-extent of the grid

        Returns:
            sdf_grid: (R, R, R) numpy array of predicted SDF
        """
        self.eval()
        with torch.no_grad():
            # Create grid
            lin = torch.linspace(-bounds, bounds, resolution)
            xx, yy, zz = torch.meshgrid(lin, lin, lin, indexing='ij')
            grid_pts = torch.stack([xx, yy, zz], dim=-1)  # (R, R, R, 3)
            grid_flat = grid_pts.reshape(1, -1, 3).to(device)  # (1, R³, 3)

            contacts = contacts.to(device)

            # Predict in chunks to avoid OOM
            chunk_size = 32768
            n_pts = grid_flat.shape[1]
            sdf_chunks = []

            latent = self.encoder(contacts)

            for i in range(0, n_pts, chunk_size):
                chunk = grid_flat[:, i:i+chunk_size, :]
                sdf_chunk = self.decoder(chunk, latent)
                sdf_chunks.append(sdf_chunk.cpu())

            sdf_flat = torch.cat(sdf_chunks, dim=1)  # (1, R³, 1)
            sdf_grid = sdf_flat.squeeze().reshape(resolution, resolution, resolution)
            return sdf_grid.numpy()

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
