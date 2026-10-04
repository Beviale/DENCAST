"""A contrastive encoder for sliding windows, to give LSH a space it can work in.

Everything measured in this project so far scored one timestamp at a time. That
throws away the thing that distinguishes an attack on a water treatment plant
from a momentary sensor blip: an attack is an *evolution* that stays wrong for
minutes. This encodes a window of W consecutive readings across all N sensors
into one vector, so the geometry downstream sees a trajectory rather than a
point.

**Why a learned space rather than the raw window.** The LSH failure mode on this
project was measured, not suspected: the cosine similarity to a point's nearest
neighbour had mean 0.99621 and standard deviation 0.00289, while the LSH returned
a maximum 0.0133 below the true one. The error was 4.6 times the width of the
entire signal, so the ranking reflected which searches had failed rather than
which points were isolated. That band is narrow by construction -- min-max
vectors all lie in the positive orthant, so their mutual cosines crowd towards
one -- and flattening a window makes it worse, since consecutive rows at 1 Hz are
nearly identical.

A contrastive objective optimises exactly the quantity that has to grow. It pulls
positives together and pushes everything else apart, which is a direct instruction
to spread the similarity distribution, and it does so in 32 or 64 dimensions
instead of W x N. Both are the conditions under which LSH stops being noise. The
check is therefore not "does the loss go down" but "did the spread widen", which
`latent_spread` reports.

**Pairs are built as specified**: a positive is an augmented view of a window
drawn within `max_offset` steps of the anchor, and negatives are the other
windows in the batch, which are sampled across the whole series and so are
distant in time. Two known hazards come with that choice and are left visible
rather than hidden, since the knobs are what control them. With W=60 at 1 Hz,
adjacent windows share 59 of 60 rows, so a positive taken at offset 1 is a 98%
copy of the anchor and the task collapses to learning the identity --
`max_offset` should be large enough that the overlap is partial. And a treatment
plant is cyclic, so two windows hours apart may be the same operating state
wrongly pushed apart as a negative; in-batch sampling accepts some of that, as
SimCLR does, in exchange for not needing labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from loguru import logger
import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    """How the series is cut into windows."""

    width: int
    """W, consecutive readings per window."""

    stride: int
    """Step between the start of one window and the next."""


def make_windows(
    X: np.ndarray, spec: WindowSpec, labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Cut a (T, N) series into (n_windows, W, N).

    Returns (windows, window_labels, end_indices). A window is labelled positive
    when *any* row inside it is, which is the convention the time-series anomaly
    literature uses: an attack that begins mid-window has already made that
    stretch of behaviour abnormal, and requiring the whole window to be attack
    would discard every onset. `end_indices` gives the row each window ends on,
    so a window can be traced back to a timestamp.
    """
    T, N = X.shape
    if spec.width > T:
        raise ValueError(f"finestra {spec.width} piu lunga della serie ({T})")
    starts = np.arange(0, T - spec.width + 1, spec.stride)
    idx = starts[:, None] + np.arange(spec.width)[None, :]
    windows = X[idx]
    win_labels = labels[idx].max(axis=1) if labels is not None else None
    return windows, win_labels, starts + spec.width - 1


def augment(
    batch: np.ndarray, rng: np.random.Generator, mask_prob: float, noise_sd: float
) -> np.ndarray:
    """An alternative view of the same windows: sensor masking plus white noise.

    Masking drops whole sensors for the whole window rather than scattered
    values. A scattered mask is trivially repaired by the neighbouring instants,
    so the encoder would learn nothing from it; losing a sensor outright forces
    it to represent the state from the others, which is the invariance wanted --
    and it is also the situation a real fault produces.
    """
    out = batch.copy()
    n, _, n_sensors = out.shape
    if mask_prob > 0:
        drop = rng.random((n, 1, n_sensors)) < mask_prob
        out = np.where(drop, 0.0, out)
    if noise_sd > 0:
        out = out + rng.normal(0.0, noise_sd, size=out.shape)
    return out.astype(np.float32)


def build_encoder(n_sensors: int, width: int, latent: int):
    """A small 1D CNN over time, with the sensors as channels.

    Convolving along time with the sensors as channels is what makes this a
    sequence model rather than an MLP on a flattened vector: a filter sees the
    same sensors at consecutive instants and can represent a rate of change,
    which is the form most of these attacks take.
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(n_sensors, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, latent),
    )


def nt_xent(z1, z2, temperature: float):
    """NT-Xent over a batch: each anchor's positive is its paired view.

    The 2B embeddings are L2-normalised and every pair scored by cosine, the
    diagonal removed so nothing is its own positive, and cross-entropy applied
    with the paired view as the target. Every other window in the batch is a
    negative, which is where the "temporally distant" requirement is satisfied:
    batches are drawn at random across the whole series.
    """
    import torch
    import torch.nn.functional as fn

    z = fn.normalize(torch.cat([z1, z2], dim=0), dim=1)
    n = z1.shape[0]
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return fn.cross_entropy(sim, targets)


def latent_spread(z: np.ndarray, sample: int = 4000, seed: int = 0) -> dict:
    """How widely the nearest-neighbour cosine is distributed in this space.

    This is the number the whole encoder exists to move. LSH is usable when its
    approximation error is small next to the spread of the signal; on the raw
    data that spread was 0.00289 and the error 0.0133, which is why the detector
    sat at the base rate. Reported before and after encoding, it says whether the
    representation fixed the problem -- independently of any detection score, and
    before the expensive parts of the pipeline are run.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z), size=min(sample, len(z)), replace=False)
    a = z[idx]
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    sims = a @ a.T
    np.fill_diagonal(sims, -np.inf)
    best = sims.max(axis=1)
    return {
        "nn_cosine_mean": float(best.mean()),
        "nn_cosine_sd": float(best.std()),
        "nn_cosine_p01": float(np.percentile(best, 1)),
        "pairwise_cosine_sd": float(sims[np.isfinite(sims)].std()),
    }


def train_encoder(
    windows: np.ndarray,
    latent: int = 32,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.1,
    max_offset: int = 5,
    mask_prob: float = 0.15,
    noise_sd: float = 0.02,
    seed: int = 42,
):
    """Train the encoder and return (model, per-epoch losses).

    `max_offset` is the positive-pair radius in window steps. At offset 0 the
    positive is only an augmented view of the anchor; larger values also treat
    nearby windows as the same state, which is what the design asks for -- but
    with a stride smaller than the width those windows overlap, so a small
    offset makes the task nearly trivial.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, width, n_sensors = windows.shape
    model = build_encoder(n_sensors, width, latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    index = torch.arange(n)
    loader = DataLoader(TensorDataset(index), batch_size=batch_size, shuffle=True, drop_last=True)
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (batch_idx,) in loader:
            i = batch_idx.numpy()
            # The positive: a window within max_offset of the anchor, clipped to
            # the series, then augmented. Both views are augmented so neither is
            # the clean original and the encoder cannot tell them apart by that.
            j = np.clip(i + rng.integers(-max_offset, max_offset + 1, size=len(i)), 0, n - 1)
            v1 = augment(windows[i], rng, mask_prob, noise_sd)
            v2 = augment(windows[j], rng, mask_prob, noise_sd)
            # (B, W, N) -> (B, N, W): convolutions run along time.
            t1 = torch.from_numpy(v1).permute(0, 2, 1)
            t2 = torch.from_numpy(v2).permute(0, 2, 1)
            loss = nt_xent(model(t1), model(t2), temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(i)
        losses.append(total / (len(loader) * batch_size))
        logger.info("  epoca {:>2}/{}  loss {:.4f}", epoch, epochs, losses[-1])
    return model, losses


def encode(model, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Project windows into the latent space, in evaluation mode."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size].astype(np.float32)
            t = torch.from_numpy(chunk).permute(0, 2, 1)
            out.append(model(t).numpy())
    return np.vstack(out)


__all__ = [
    "WindowSpec",
    "make_windows",
    "augment",
    "build_encoder",
    "nt_xent",
    "latent_spread",
    "train_encoder",
    "encode",
]
