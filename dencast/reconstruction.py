"""A convolutional autoencoder over windows, scored by reconstruction error.

The step before a GM-VAE, and the one that decides whether building it is worth
the trouble. A mixture prior adds multi-modality so a density clustering has
somewhere to live; it does not, by itself, make reconstruction a good detector.
If the plain reconstruction error cannot beat the numbers already measured,
nothing built on the same principle will.

**Why reconstruction rather than the contrastive objective that failed.** The
contrastive encoder placed attack windows *inside* the normal clusters -- exact
1-NN in its latent space scored ROC-AUC 0.4153, below chance, i.e. inverted. The
diagnosis was that a contrastive loss arranges what it sees and nothing
constrains where unseen input lands. Asking a decoder to redraw the window
constrains it: the latent has to carry enough to reproduce the dynamics, so a
window the model has no account of cannot be represented cheaply.

**The latent width is the parameter that decides this, not a detail.** A
bottleneck wide enough to copy its input reconstructs anomalies as faithfully as
normal data and the score goes flat -- the documented failure of deep generative
models on out-of-distribution input. Too narrow and normal windows reconstruct
badly too, and the error stops separating anything. `train_autoencoder` is meant
to be swept over it.

**The frozen-sensor case, which is the reason for the whole approach.** Many
SWaT attacks hold a sensor at a fixed value. Pointwise that is just a value; it
is only anomalous as a *dynamic*, which is what a window shows. If the latent
encodes "the plant is in operating state A", the decoder redraws the oscillation
that state A implies, and the flat input disagrees with it -- large error on that
one sensor, which is both the detection and the explanation.
"""

from __future__ import annotations

from typing import List, Tuple

from loguru import logger
import numpy as np


def _encoder_length(width: int) -> int:
    """Time steps left after the two stride-2 convolutions."""
    after_first = (width + 2 * 2 - 5) // 2 + 1
    return (after_first + 2 * 1 - 3) // 2 + 1


class _CropTime(__import__("torch").nn.Module):
    """Trim the time axis back to exactly `width`.

    The two stride-2 convolutions halve the length twice, so the transposed ones
    return ceil(width / 4) * 4 -- exact only when the width is a multiple of
    four. Without this, W=30 reconstructs to 32 and the MSE is computed against
    a mismatched tensor, which either raises or, worse, broadcasts silently.
    """

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, x):
        return x[..., : self.width]


def build_autoencoder(n_sensors: int, width: int, latent: int):
    """Encoder and decoder over (N channels, W time steps).

    Convolutions run along time with the sensors as channels, so a filter sees
    the same instruments at consecutive instants and can represent a rate of
    change -- the form most of these attacks take. The decoder mirrors it with
    transposed convolutions back to the original W.
    """
    import torch.nn as nn

    enc_len = _encoder_length(width)
    encoder = nn.Sequential(
        nn.Conv1d(n_sensors, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * enc_len, latent),
    )
    decoder = nn.Sequential(
        nn.Linear(latent, 128 * enc_len),
        nn.ReLU(),
        nn.Unflatten(1, (128, enc_len)),
        nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.ConvTranspose1d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, n_sensors, kernel_size=5, padding=2),
        _CropTime(width),
    )
    return encoder, decoder


class AutoEncoder:
    """Encoder plus decoder, kept together so the pair cannot drift apart."""

    def __init__(self, n_sensors: int, width: int, latent: int):
        import torch.nn as nn

        enc, dec = build_autoencoder(n_sensors, width, latent)
        self.model = nn.Sequential(enc, dec)
        self.encoder, self.decoder = enc, dec
        self.width, self.n_sensors, self.latent = width, n_sensors, latent


def train_autoencoder(
    windows: np.ndarray,
    latent: int = 16,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple[AutoEncoder, List[float]]:
    """Fit on normal windows only; returns the model and the per-epoch MSE.

    The decoder's output is compared against the *unaugmented* input: there is no
    corruption step here, unlike the contrastive training. A denoising objective
    would teach the model to repair exactly the kind of local damage an attack
    produces, which is the opposite of what is wanted.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    n, width, n_sensors = windows.shape
    ae = AutoEncoder(n_sensors, width, latent)
    opt = torch.optim.Adam(ae.model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # (B, W, N) -> (B, N, W) once, rather than per batch.
    tensor = torch.from_numpy(windows.astype(np.float32)).permute(0, 2, 1)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

    losses: List[float] = []
    for epoch in range(1, epochs + 1):
        ae.model.train()
        total, seen = 0.0, 0
        for (batch,) in loader:
            out = ae.model(batch)
            loss = loss_fn(out, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(batch)
            seen += len(batch)
        losses.append(total / seen)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            logger.info("    epoca {:>2}/{}  MSE {:.6f}", epoch, epochs, losses[-1])
    return ae, losses


def reconstruction_error(
    ae: AutoEncoder, windows: np.ndarray, batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-window score and per-sensor error.

    Returns (score, per_sensor) where `score` is the mean squared error over the
    whole window and `per_sensor` is (n_windows, n_sensors), averaged over time.
    The second is the attribution: which instrument the model could not account
    for, available for free from the same forward pass rather than from a
    separate explanation method.
    """
    import torch

    ae.model.eval()
    scores, per_sensor = [], []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size].astype(np.float32)
            x = torch.from_numpy(chunk).permute(0, 2, 1)
            err = (ae.model(x) - x) ** 2
            per_sensor.append(err.mean(dim=2).numpy())
            scores.append(err.mean(dim=(1, 2)).numpy())
    return np.concatenate(scores), np.vstack(per_sensor)


def encode(ae: AutoEncoder, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Latent vectors, for the clustering stages that come after."""
    import torch

    ae.model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size].astype(np.float32)
            out.append(ae.encoder(torch.from_numpy(chunk).permute(0, 2, 1)).numpy())
    return np.vstack(out)


__all__ = [
    "AutoEncoder",
    "build_autoencoder",
    "train_autoencoder",
    "reconstruction_error",
    "encode",
]
