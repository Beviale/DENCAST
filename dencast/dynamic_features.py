"""Compact dynamic features: what a window *did*, not every value it held.

A window of W steps over N sensors flattened is W x N numbers, most of them
redundant -- at 1 Hz consecutive rows barely differ. Measured on SWaT, that
redundancy costs accuracy: flattened windows of 60 steps scored ROC-AUC 0.8790
where the same detector pointwise scored 0.9029, because an attack touching one
sensor for part of the window is a handful of 2,640 values and a distance over
all of them averages it away.

Three statistics per sensor replace the concatenation:

    media       where the sensor sat
    dev.std     how much it moved
    derivata    the mean absolute step between consecutive instants

The second and third are the point. A frozen sensor -- the most common SWaT
attack -- has a normal mean and a standard deviation and a derivative of nearly
zero, so it is anomalous in exactly the coordinates a concatenation buries. And
3N numbers instead of W x N removes the redundancy without removing the dynamics.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def dynamic_features(windows: np.ndarray, with_range: bool = False) -> np.ndarray:
    """(n_windows, W, N) -> (n_windows, 3N or 4N) per sensor.

    Mean, standard deviation and mean absolute step; with `with_range`, also the
    max minus min. The range is largely redundant with the standard deviation --
    a frozen sensor already shows zero in both, and in the derivative too -- and
    earns its place only in one case: a sensor that is nearly flat but carries a
    single spike, where the standard deviation stays moderate while the range
    does not.

    The three blocks are laid out contiguously -- all means, then all standard
    deviations, then all derivatives -- so a coordinate can be traced back to a
    sensor and a statistic by integer division, which is what the attribution
    step needs.
    """
    if windows.ndim != 3:
        raise ValueError(f"attese finestre (n, W, N), ricevuto {windows.shape}")
    mean = windows.mean(axis=1)
    sd = windows.std(axis=1)
    # Mean absolute first difference: movement, independent of direction. A
    # window whose sensor is held constant has zero here whatever its level.
    deriv = np.abs(np.diff(windows, axis=1)).mean(axis=1) if windows.shape[1] > 1 \
        else np.zeros_like(mean)
    blocks = [mean, sd, deriv]
    if with_range:
        blocks.append(windows.max(axis=1) - windows.min(axis=1))
    return np.hstack(blocks).astype(np.float32)


def feature_names(sensors: List[str], with_range: bool = False) -> List[str]:
    """Names matching the layout of `dynamic_features`."""
    names = (
        [f"{s}|media" for s in sensors]
        + [f"{s}|sd" for s in sensors]
        + [f"{s}|deriv" for s in sensors]
    )
    if with_range:
        names += [f"{s}|range" for s in sensors]
    return names


def sensor_of(index: int, n_sensors: int) -> Tuple[int, str]:
    """Which sensor and which statistic a feature coordinate belongs to."""
    block = index // n_sensors
    # Named blocks first, then whatever a caller appended -- conditional
    # residuals add two more, and an unknown block must not raise: the
    # attribution runs after the detection and crashing there would discard a
    # completed run over a label.
    names = ("media", "sd", "deriv", "range", "res_media", "res_max")
    return index % n_sensors, names[block] if block < len(names) else f"blocco{block}"


__all__ = ["dynamic_features", "feature_names", "sensor_of"]
