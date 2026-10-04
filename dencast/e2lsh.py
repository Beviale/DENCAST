"""Euclidean LSH: hashing by distance instead of by direction.

DENCAST hashes with random hyperplanes, which is a *cosine* scheme: it reads the
direction of a vector and is blind to its length. That is not a detail on
standardised features -- it was measured. The same 132 dynamic features scored
ROC-AUC 0.8942 under a Euclidean nearest-neighbour distance and 0.5185 once
projected onto the unit sphere, because the norm of a standardised vector *is*
the anomaly signal: it says how far the window sits from the training mean
overall. Normalising discards exactly that.

Appending a large constant before normalising recovers it, but only by trading
away the other thing the index needs. Measured across constants, the detection
score rises to 0.8934 while the spread of the nearest-neighbour cosine collapses
from 0.029 to 0.007 and below -- and a narrow spread is what made the
hyperplane LSH useless in the first place, since its approximation error then
exceeds the whole signal. The two requirements are in direct conflict under a
cosine index.

E2LSH removes the conflict rather than balancing it. With p-stable (Gaussian)
projections,

    h(v) = floor((a . v + b) / w),    a ~ N(0, I),   b ~ U[0, w]

the probability that two points collide falls with their Euclidean distance, so
the index groups by distance and nothing has to be normalised away.

**Bands are an AND of hashes, ORed across bands.** Within a band, `n_hashes`
values must match exactly, which makes a collision demanding; across `n_bands`
independent bands, any single match proposes the pair. That is the usual knob
pair: more hashes per band raises precision, more bands raises recall, and both
cost time linearly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from loguru import logger
import numpy as np


def hash_codes(
    X: np.ndarray, n_hashes: int, n_bands: int, width: float, seed: int
) -> np.ndarray:
    """(n, n_bands, n_hashes) integer codes.

    All bands are projected in one matrix multiply and then reshaped; drawing
    them band by band would be the same arithmetic in a slower loop.
    """
    rng = np.random.default_rng(seed)
    total = n_hashes * n_bands
    a = rng.standard_normal((X.shape[1], total))
    b = rng.uniform(0.0, width, size=total)
    projected = (X @ a + b) / width
    return np.floor(projected).astype(np.int64).reshape(len(X), n_bands, n_hashes)


def _band_groups(codes_band: np.ndarray) -> Dict[bytes, List[int]]:
    """Objects sharing an identical code tuple in one band."""
    groups: Dict[bytes, List[int]] = {}
    for i, row in enumerate(codes_band):
        groups.setdefault(row.tobytes(), []).append(i)
    return groups


def candidate_pairs(
    X: np.ndarray,
    n_hashes: int = 4,
    n_bands: int = 20,
    width: Optional[float] = None,
    radius: Optional[float] = None,
    seed: int = 42,
    max_bucket: int = 400,
) -> set:
    """Candidate (i, j) pairs, i < j, from the OR over bands.

    `width` defaults to twice the radius, which was measured rather than
    reasoned. At width equal to the radius, two points exactly R apart rarely
    land in the same interval on all `n_hashes` projections at once, and recall
    came out at 40.8%. Doubling it takes recall to 97.3% for about four times the
    candidates, and doubling again reaches 100% for a further 17% -- so twice is
    where the curve turns.

    `max_bucket` caps how many objects one bucket may contribute pairs from. A
    degenerate bucket holding thousands of near-identical windows -- which 1 Hz
    data produces readily -- would otherwise generate millions of pairs and undo
    the point of indexing. Oversized buckets are sampled rather than dropped, so
    they still contribute.
    """
    if width is None:
        width = 2.0 * radius if radius else 1.0
    codes = hash_codes(X, n_hashes, n_bands, width, seed)
    rng = np.random.default_rng(seed + 7)
    pairs: set = set()
    n_capped = 0
    for band in range(n_bands):
        for members in _band_groups(codes[:, band, :]).values():
            if len(members) < 2:
                continue
            if len(members) > max_bucket:
                n_capped += 1
                members = rng.choice(members, size=max_bucket, replace=False).tolist()
            arr = np.array(sorted(members))
            iu, ju = np.triu_indices(len(arr), k=1)
            pairs.update(zip(arr[iu].tolist(), arr[ju].tolist()))
    if n_capped:
        logger.info("  {} bucket oltre {} membri, campionati", n_capped, max_bucket)
    return pairs


def cross_candidates(
    Q: np.ndarray,
    R: np.ndarray,
    n_hashes: int = 4,
    n_bands: int = 20,
    width: Optional[float] = None,
    radius: Optional[float] = None,
    seed: int = 42,
    max_bucket: int = 400,
) -> Dict[int, List[int]]:
    """For each query row, the reference rows it collides with.

    Both sides are hashed with the *same* projections -- they are drawn once from
    the pooled matrix -- because codes from different projections are not
    comparable and the mistake would be silent: buckets would still form and
    still produce pairs, just meaningless ones.
    """
    if width is None:
        width = 2.0 * radius if radius else 1.0
    pooled = np.vstack([R, Q])
    codes = hash_codes(pooled, n_hashes, n_bands, width, seed)
    n_ref = len(R)
    rng = np.random.default_rng(seed + 7)
    out: Dict[int, List[int]] = {}
    for band in range(n_bands):
        for members in _band_groups(codes[:, band, :]).values():
            refs = [m for m in members if m < n_ref]
            queries = [m - n_ref for m in members if m >= n_ref]
            if not refs or not queries:
                continue
            if len(refs) > max_bucket:
                refs = rng.choice(refs, size=max_bucket, replace=False).tolist()
            for q in queries:
                out.setdefault(q, []).extend(refs)
    return {q: sorted(set(v)) for q, v in out.items()}


def choose_radius(
    X: np.ndarray, min_pts: int, quantile: float = 0.95, sample: int = 4000, seed: int = 0
) -> Tuple[float, dict]:
    """A radius that leaves a chosen fraction of the training set isolated.

    Picked from the distance to the `min_pts`-th neighbour rather than to the
    first: an object is core when it has min_pts neighbours inside the radius, so
    that is the distance the threshold has to clear. Setting it at the 95th
    percentile means about 5% of normal training windows fall short and become
    noise -- a physiological minority, which is what the design wants and what
    every cosine attempt failed to produce, landing at 0% or 68% instead.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    k = min(min_pts + 1, len(X) - 1)
    d, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X[idx])
    kth = d[:, -1]
    radius = float(np.quantile(kth, quantile))
    return radius, {
        "kth_nn_median": float(np.median(kth)),
        "kth_nn_p95": float(np.quantile(kth, 0.95)),
        "first_nn_median": float(np.median(d[:, 1])) if d.shape[1] > 1 else 0.0,
        "radius": radius,
        "quantile": quantile,
        "min_pts": min_pts,
    }


__all__ = ["hash_codes", "candidate_pairs", "cross_candidates", "choose_radius"]
