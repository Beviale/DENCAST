"""Calibrating the expected error against the confidence of the assignment.

DENCAST routes an object to a cluster by finding its most similar labeled
object (Eq. 1 of the paper). How similar that neighbour is says a lot about
how much error to expect afterwards:

    nearest neighbour very close  ->  a small residual is normal,
                                      a large one is surprising
    nearest neighbour far away    ->  a large residual is expected,
                                      and carries no information

So the interesting quantity is not the residual but how far it exceeds what is
normal at that level of similarity. This module estimates that "normal", as a
curve g, and the anomaly score divides by it.

**Why the similarity has to be transformed.** Measured on PV Italy, the cosine
between an object and its nearest neighbour saturates: 85% of test objects sit
above 0.988, and the whole discriminating range is in the last three
thousandths. Fitted against the raw similarity the curve has nothing to grip --
the linear correlation with the residual came out at +0.04, essentially zero
and with the wrong sign. Against -log(1 - sim) the tail is stretched
(0.99 -> 4.6, 0.999 -> 6.9, 0.9999 -> 9.2) and the structure appears: over
three days the median residual falls from ~0.05 to 0.000 across the top bins.

**Why monotonicity is imposed rather than hoped for.** The raw binned means are
noisy and not decreasing -- 0.058, 0.063, 0.078, 0.056, 0.028, 0.020 on those
same three days. Isotonic regression fits the closest non-increasing curve,
which is the one shape assumption worth making here, and adds no weights or
functional form of its own.
"""

from __future__ import annotations

from typing import List, Sequence

from loguru import logger
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

MIN_EXPECTED_RESIDUAL = 1e-4
"""Floor on g.

The empirical expected residual reaches exactly 0 in the top similarity bin --
those objects have a neighbour carrying the same target to the digit -- and
dividing by it would make every one of them infinitely anomalous. The floor is
in the units of the normalised targets, so roughly a ten-thousandth of range.
"""


def similarity_to_distance(sim_col: Column, eps: float = 1e-9) -> Column:
    """Map a cosine similarity to -log(1 - sim), stretching the saturated tail."""
    return -F.log(F.greatest(F.lit(1.0) - sim_col, F.lit(eps)))


def pava_non_increasing(
    values: Sequence[float], weights: Sequence[float]
) -> List[float]:
    """Pool-adjacent-violators for a non-increasing fit.

    Walks left to right; wherever a value is higher than the one before it --
    which a decreasing curve forbids -- the two blocks merge into their
    weighted mean, and the walk steps back because a merge can break the
    ordering behind it. The result is the closest non-increasing sequence in
    weighted least squares, i.e. isotonic regression.

    Run on the driver on purpose: the input is one row per bin, twenty of them,
    so handing it to Spark ML would cost more than it saves.

    Returns one value per input element, each being the value of the block it
    ended up in.
    """
    if not values:
        return []

    # Each block keeps its own weight and how many original bins it covers.
    blocks: List[List[float]] = [[v, w, 1] for v, w in zip(values, weights)]

    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            v1, w1, c1 = blocks[i]
            v2, w2, c2 = blocks[i + 1]
            total = w1 + w2
            merged = [(v1 * w1 + v2 * w2) / total if total else v1, total, c1 + c2]
            blocks[i : i + 2] = [merged]
            i = max(i - 1, 0)
        else:
            i += 1

    out: List[float] = []
    for value, _, count in blocks:
        out.extend([value] * int(count))
    return out


class ExpectedResidual:
    """g as a piecewise-constant, non-increasing curve over -log(1 - sim)."""

    def __init__(
        self,
        upper_bounds: Sequence[float],
        values: Sequence[float],
        floor: float = MIN_EXPECTED_RESIDUAL,
    ):
        self.upper_bounds = list(upper_bounds)
        self.values = [max(float(v), floor) for v in values]
        self.floor = floor

    def apply(self, distance_col: Column) -> Column:
        """Build the lookup as a Spark expression, no UDF involved."""
        expr = F.lit(self.values[-1])
        # Backwards, so the first matching bound wins.
        for bound, value in zip(
            reversed(self.upper_bounds), reversed(self.values)
        ):
            expr = F.when(distance_col <= F.lit(float(bound)), F.lit(value)).otherwise(
                expr
            )
        return expr

    def as_dict(self) -> dict:
        """Plain representation, for logging or for saving beside the model."""
        return {
            "upper_bounds": self.upper_bounds,
            "values": self.values,
            "floor": self.floor,
        }

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"d<={b:.2f}: {v:.4f}" for b, v in zip(self.upper_bounds, self.values)
        )
        return f"ExpectedResidual({pairs})"


def fit_expected_residual(
    calibration: DataFrame,
    n_bins: int = 20,
    floor: float = MIN_EXPECTED_RESIDUAL,
) -> ExpectedResidual:
    """Estimate g from (best_sim, residual) pairs.

    Args:
        calibration: needs `best_sim` and `residual`, and those pairs must come
            from objects that were **not** in the training set of the model
            that produced them, and that are disjoint from the objects the
            curve will later score. `collect_calibration_pairs` builds them
            that way. Calibrating on the objects being scored would let the
            anomalies raise the very baseline they are meant to stand out
            from -- the threshold would quietly move to accommodate them.
        n_bins: quantile bins over the transformed similarity. More bins give a
            finer and noisier curve; the isotonic step absorbs part of that by
            merging neighbours that violate the ordering.
        floor: minimum value, see MIN_EXPECTED_RESIDUAL.

    The bin summary is the median rather than the mean: the residual
    distribution has a long right tail, and a handful of genuinely anomalous
    objects in the calibration set would otherwise raise the very baseline
    they are supposed to stand out from.
    """
    with_d = calibration.withColumn(
        "d", similarity_to_distance(F.col("best_sim"))
    ).cache()

    probs = [i / n_bins for i in range(1, n_bins)]
    edges = sorted({round(e, 6) for e in with_d.approxQuantile("d", probs, 0.001)})
    if not edges:
        with_d.unpersist()
        raise ValueError(
            "best_sim has no usable spread: g cannot be calibrated. "
            "This happens when every object has an almost identical neighbour."
        )

    bucket = F.lit(len(edges))
    for i, edge in enumerate(reversed(edges)):
        bucket = F.when(
            F.col("d") <= F.lit(edge), F.lit(len(edges) - 1 - i)
        ).otherwise(bucket)

    binned = (
        with_d.withColumn("bin", bucket)
        .groupBy("bin")
        .agg(
            F.max("d").alias("upper"),
            F.expr("percentile_approx(residual, 0.5)").alias("median_residual"),
            F.count("*").alias("n"),
        )
        .orderBy("bin")
        .collect()
    )
    with_d.unpersist()

    uppers = [float(r["upper"]) for r in binned]
    medians = [float(r["median_residual"] or 0.0) for r in binned]
    counts = [float(r["n"]) for r in binned]

    fitted = pava_non_increasing(medians, counts)

    n_merged = len(medians) - len(set(fitted))
    logger.info(
        "Calibrated g over {} bins ({} merged by the monotonicity constraint); "
        "expected residual falls from {:.4f} to {:.4f}",
        len(fitted),
        n_merged,
        fitted[0],
        fitted[-1],
    )
    return ExpectedResidual(uppers, fitted, floor)

# ---------------------------------------------------------------------------
# Building the calibration set inductively
# ---------------------------------------------------------------------------


def collect_calibration_pairs(spark, df, params, dates, seed: int) -> DataFrame:
    """Gather (best_sim, residual) pairs the model has never seen.

    For each calibration day the model is fit on the window *before* it and
    then predicts that day, so every pair comes from an object outside the
    training set that produced it. That is what makes the curve inductive:
    fitting g on the objects it will later score would let the anomalies lift
    the baseline they are supposed to exceed, and the score would silently
    normalise them away.

    The days used here must also be disjoint from the days being scored --
    take them from an earlier period, or from a different split of the paper's
    date files.

    g is a calibration curve, not a model: it describes how the error decays
    with assignment confidence, which changes slowly. One fit over a handful of
    days can be reused across the whole evaluation instead of being redone for
    each, which is what keeps the cost of being inductive down to a few extra
    model fits rather than one per scored day.

    Args:
        spark: the session.
        df: the processed dataset, (id, features, targets, date, ...).
        params: project parameters; `evaluation.window_size` sets the window.
        dates: the calibration days.
        seed: LSH seed, kept the same as the scoring run so the curve describes
            the same graph construction.

    Returns:
        (best_sim, residual) -- the residual summed over the k targets.
    """
    # Imported here: features and modeling import this module, and a top-level
    # import would close the cycle.
    from dencast.data.features import hide_targets, temporal_split
    from dencast.modeling.fit import fit
    from dencast.modeling.predict import attach_ground_truth, predict

    collected = None
    for date in dates:
        train, test = temporal_split(df, date, params.evaluation.window_size)
        n_train, n_test = train.count(), test.count()
        if n_train == 0 or n_test == 0:
            logger.warning("Calibration day {} has an empty split, skipped", date)
            continue

        train = train.repartition(params.spark.num_partitions).cache()
        test = test.cache()
        model = fit(spark, train, params, seed)
        scored = attach_ground_truth(
            predict(hide_targets(test), model.model, params.dataset.k), test
        )

        pairs = scored.select(
            "best_sim",
            F.aggregate(
                F.zip_with(
                    F.col("actual"), F.col("prediction"), lambda a, b: F.abs(a - b)
                ),
                F.lit(0.0),
                lambda acc, x: acc + x,
            ).alias("residual"),
        )
        collected = pairs if collected is None else collected.unionByName(pairs)
        train.unpersist()
        test.unpersist()

    if collected is None:
        raise ValueError("No usable calibration day: every split came out empty")
    return collected


def save_expected_residual(curve: ExpectedResidual, path) -> None:
    """Write the curve to JSON, so scoring runs can reuse one calibration."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(curve.as_dict(), indent=2), encoding="utf-8")
    logger.info("Calibration curve written to {}", path)


def load_expected_residual(path) -> ExpectedResidual:
    """Read a curve written by `save_expected_residual`."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExpectedResidual(
        data["upper_bounds"], data["values"], data.get("floor", MIN_EXPECTED_RESIDUAL)
    )
