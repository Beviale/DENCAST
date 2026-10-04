"""How the sensors move together: the part univariate features cannot see.

Every representation tried so far is per-sensor. Mean, standard deviation and
derivative each describe one instrument in isolation, so a fault that consists of
an impossible *combination* of individually ordinary readings -- a pump reporting
ON while its downstream flow reads zero -- is invisible by construction. In a
water treatment plant, where level, flow and pump state are bound by
conservation, that is not an edge case.

Two ways to capture it, with very different statistical footing.

**Per-window correlations** are the literal reading: compute the Pearson
correlation of each sensor pair inside the window. The problem is the sample
size. At W=10 a correlation is estimated from ten points and its standard error
is about 1/sqrt(W-3) = 0.38, so most of what it measures is noise -- and there
are 44*43/2 = 946 of them, eight times the current feature count. Restricting to
the pairs that are strongly coupled in the training data keeps the count
manageable, but not the variance.

**Conditional residuals** use the global structure instead. The precision matrix
is estimated once over the whole training split -- 496,800 rows, so it is stable
-- and then applied row by row:

    residual_j = (Theta @ x)_j / sqrt(Theta_jj)

which is the standardised residual of sensor j given every other sensor. It
answers the same question -- is this value consistent with what the others are
doing -- without estimating anything from ten points, and it is one number per
sensor, so it doubles as attribution.

Ledoit-Wolf shrinkage is used for the covariance: 44 dimensions from a long but
autocorrelated series leaves the unregularised inverse dominated by whichever
direction happened to be thinnest.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def fit_precision(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Global mean and precision matrix from the training rows."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(rows)
    return lw.location_, lw.precision_


def conditional_residuals(rows: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """(n_rows, N) standardised residual of each sensor given the others.

    Large where a reading cannot be reconciled with the rest of the plant at that
    instant, whatever its own marginal distribution says.
    """
    centred = rows - mean
    z = centred @ precision
    scale = np.sqrt(np.maximum(np.diag(precision), 1e-12))
    return z / scale


def residual_features(
    windows: np.ndarray,
    mean: np.ndarray,
    precision: np.ndarray,
    clip: Optional[float] = None,
) -> np.ndarray:
    """(n_windows, 2N): mean and max absolute conditional residual per sensor.

    Both are kept because they answer different questions: the mean says the
    sensor was inconsistent throughout the window, the maximum that it was
    inconsistent at some instant. A brief injection shows in one and not the
    other.

    `clip` caps the absolute residual, and it exists because of a measured
    pathology. Several SWaT actuators are *constant* through the whole normal
    period -- P102 never runs in 496,800 rows -- so their conditional variance is
    zero and only Ledoit-Wolf shrinkage keeps the precision matrix invertible.
    The resulting conditional standard deviation is the shrinkage floor, 0.002239,
    so any unit deviation yields 1/0.002239 = 446.5 exactly, the same value every
    time. That number measures a regularisation parameter, not the size of an
    anomaly.

    Left uncapped it owns the Euclidean distance: 446.5 squared is 199,400 against
    roughly 5,500 for all 219 other coordinates together, a factor of 36. Capping
    at 10 brings it to 100, which is 1.8% of the rest. The cost is that
    "impossible" and "ten sigma" stop being distinguishable -- and the impossible
    actuator state is a 97%-precision signal, so the cap removes the margin that
    keeps the top of the ranking clean. Whether that trade pays is measured, not
    argued.
    """
    n_win, w, n_sensors = windows.shape
    flat = windows.reshape(-1, n_sensors)
    res = np.abs(conditional_residuals(flat, mean, precision))
    if clip is not None:
        res = np.minimum(res, clip)
    res = res.reshape(n_win, w, n_sensors)
    return np.hstack([res.mean(axis=1), res.max(axis=1)]).astype(np.float32)


def top_pairs(rows: np.ndarray, n_pairs: int = 200) -> List[Tuple[int, int]]:
    """The sensor pairs most strongly correlated over the training rows.

    Correlations are computed once, globally, purely to *choose* which pairs are
    worth tracking; the per-window values are what the features carry. Selecting
    on the training data only keeps the choice out of the test split.
    """
    corr = np.corrcoef(rows, rowvar=False)
    corr = np.nan_to_num(corr)
    n = corr.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(-np.abs(corr[iu, ju]))[:n_pairs]
    return [(int(iu[o]), int(ju[o])) for o in order]


def correlation_features(
    windows: np.ndarray, pairs: List[Tuple[int, int]]
) -> np.ndarray:
    """(n_windows, len(pairs)): Pearson correlation of each pair inside the window.

    Computed by hand rather than with np.corrcoef per window, which would be a
    Python loop over thousands of windows. A pair whose sensor is constant across
    the window has no defined correlation; those come back as zero, which reads as
    "no relationship observed" and is the safe value here -- a frozen sensor is
    already described by its own standard deviation and derivative.
    """
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    xa = windows[:, :, a]
    xb = windows[:, :, b]
    xa = xa - xa.mean(axis=1, keepdims=True)
    xb = xb - xb.mean(axis=1, keepdims=True)
    num = (xa * xb).sum(axis=1)
    den = np.sqrt((xa**2).sum(axis=1) * (xb**2).sum(axis=1))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12).astype(np.float32)


def coupling_names(
    sensors: List[str],
    mode: str,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> List[str]:
    """Names matching the layout each mode produces."""
    if mode == "resid":
        return ([f"{s}|res_media" for s in sensors]
                + [f"{s}|res_max" for s in sensors])
    if mode == "corr" and pairs is not None:
        return [f"{sensors[i]}~{sensors[j]}" for i, j in pairs]
    return []


__all__ = [
    "fit_precision",
    "conditional_residuals",
    "residual_features",
    "top_pairs",
    "correlation_features",
    "coupling_names",
]


def constant_columns(rows: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Indices of columns that never vary over the training rows.

    They are the source of the 446.5 artefact: zero variance means the
    conditional standard deviation falls back to the Ledoit-Wolf shrinkage floor,
    and any deviation then divides by it.

    **Dropping them is not automatically right**, which is why this only reports
    them. On SWaT they split into two kinds that a training-only test cannot
    separate: five are constant in the test period too and carry nothing, while
    `P102` and `P206` change during attacks and are the most precise signal in the
    dataset (97.0% precision at 5.6% recall). Removing the second kind to be rid
    of the first throws away the result.
    """
    return np.flatnonzero(rows.std(axis=0) <= tol)


def novelty_flags(
    windows: np.ndarray, reference: np.ndarray, columns: np.ndarray, scale: float = 10.0
) -> np.ndarray:
    """(n_windows, len(columns)): did this window ever leave the training constant?

    The honest encoding of what a constant column can tell us. Instead of asking a
    Gaussian precision matrix to express "never seen before" -- which it does only
    through the reciprocal of a regularisation parameter -- this states it
    directly: `scale` when any instant in the window differs from the value the
    column held throughout training, zero otherwise.

    `scale` puts it on the same footing as the other coordinates rather than 36
    times above them, which is the whole point.
    """
    if len(columns) == 0:
        return np.zeros((len(windows), 0), dtype=np.float32)
    differs = np.abs(windows[:, :, columns] - reference[columns]) > 1e-9
    return (differs.any(axis=1) * scale).astype(np.float32)


def discrete_columns(rows: np.ndarray, max_levels: int = 5) -> np.ndarray:
    """Indices of columns taking at most `max_levels` distinct training values.

    On SWaT this is 20 of 44: the pumps, the motorised valves and the UV lamp.
    They are states, not measurements, and forcing them through statistics built
    for continuous signals is what produced the shrinkage artefact in the first
    place.
    """
    return np.flatnonzero(
        np.array([len(np.unique(rows[:, j])) <= max_levels for j in range(rows.shape[1])])
    )


def mismatch_rates(
    windows: np.ndarray, modes: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    """(n_windows, len(columns)): fraction of instants away from the training mode.

    The graded form of "this state is wrong", and the thing a binary per-window
    flag could not express. Measured, that flag recovered only part of what the
    clipped Gaussian residual achieved -- F1 0.8098 against 0.8475 -- because over
    ten instants a pump on for one second and on for ten are different events and
    one bit cannot say which. A rate in [0, 1] can.

    The mode comes from the training rows, so a column that legitimately switches
    during normal operation gets a non-trivial baseline rather than being treated
    as constant.
    """
    if len(columns) == 0:
        return np.zeros((len(windows), 0), dtype=np.float32)
    differs = windows[:, :, columns] != modes[columns]
    return differs.mean(axis=1).astype(np.float32)


def training_modes(rows: np.ndarray) -> np.ndarray:
    """The most frequent value of each column over the training rows."""
    out = np.empty(rows.shape[1])
    for j in range(rows.shape[1]):
        vals, counts = np.unique(rows[:, j], return_counts=True)
        out[j] = vals[int(np.argmax(counts))]
    return out


def scale_block(train_block: np.ndarray, test_block: np.ndarray, weight: float):
    """Put a block on unit mean squared norm, then apply its weight.

    Normalising first is what makes the weight mean something: without it the
    weight competes with however many columns the block happens to have and
    whatever scale they arrived on, and "equal weight" would not be equal. With
    it, weight 1 on both blocks means both contribute the same amount to the
    squared distance on average, and any departure from that is a choice someone
    made rather than an accident of construction.
    """
    norm = float(np.sqrt((train_block**2).sum(axis=1).mean()))
    if norm <= 1e-12:
        norm = 1.0
    factor = weight / norm
    return train_block * factor, test_block * factor
