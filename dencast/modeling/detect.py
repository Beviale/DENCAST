"""Scoring a split for anomalies: what `train_and_select` and `train_winner` share.

One model, fitted once on a whole split, then used to score another. No rolling
window: the split is chronological, so everything in train already precedes
everything in valid, and refitting per day would throw away five hundred days of
history to honour a constraint the ordering already guarantees. It is also far
cheaper -- one fit per grid point instead of one per scored day.

**Routing goes through LSH blocking, not centroids.** Comparing each object to
be scored against every training object is exact and quadratic: 43,000 x 162,000
is seven billion cosines. Comparing against one mean vector per cluster is cheap
and wrong for this algorithm -- a density-based cluster may be long or curved and
its mean can fall outside it, so centroid routing would turn inference into
K-means and throw away the property that motivated DENCAST. LSH blocking gives a
constant number of *real* candidate nodes per object, which keeps arbitrary
shapes and introduces no error the graph does not already accept.

Four rankings come back from every run:

    sigmoid_plain   sigmoid((mean_z2 - offset) / scale)
    sigmoid_sim     sigmoid((mean_z2 * best_sim - offset) / scale)
    sum_sq          mean_z2 itself, untransformed
    raw_sim         mean_z2 * best_sim, untransformed

The last two are not extras. A sigmoid is monotonic, so `sigmoid_plain` must
rank exactly as `sum_sq` does -- if they ever disagree, the sigmoid has saturated
and float64 has collapsed distinct scores into ties. Reporting both turns that
failure from invisible into obvious.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence, Tuple

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from dencast.deviation import (
    build_routing_index,
    column_moments,
    column_quantiles,
    deviation_scores,
    quantile_scores,
    route_batch,
    sigmoid_scores,
)
from dencast.modeling.fit import DencastModel, fit
from dencast.utils import Params

FORMULAS = (
    "sigmoid_plain",
    "sigmoid_sim",
    "sum_sq",
    "raw_sim",
    "score_top1",
    "score_top3",
    "score_top5",
    "q_max",
    "q_top3",
    "q_sum",
    "iso_nn",
    "iso_degree",
    "iso_knn",
)
"""The rankings every run reports.

The first four are the sigmoid formulas and their untransformed counterparts,
all built on `mean_z2` -- the sum of squared per-column z over every column. The
last three are the same deviations aggregated over the k largest only, which
`deviation_scores` already computes.

They are here because the aggregation turned out to matter more than anything
else measured. On identical days, with a global reference: the full sum scores
AUC-PR 0.0157, the maximum 0.0351, the top-3 0.0565. Putting the cluster
reference under the full sum lifts it to 0.0232 -- a real gain, and still less
than what changing the aggregation alone buys. Reporting only the sum would have
hidden that the weakest part of the formula was the part nobody was varying.

The `q_*` three replace the z entirely with an empirical rank: how deep in its
cluster's own distribution for that column the value sits, both tails counted.
They exist because a z-score assumes a shape these columns do not have. `kwh`
and `irradiamento` are strongly skewed with a mass at zero, and the failure is
concrete rather than theoretical -- an injected `drop` moves the target *toward*
the mean, so its z falls and a z-based score calls it more normal than the value
it replaced. Measured globally, drops came out at ROC-AUC 0.41, below chance. A
rank has no such blind spot and assumes nothing about the distribution.

The `iso_*` three do not decompose by column at all. They ask only how isolated
the object is in the joint space:

    iso_nn      1 - similarity to the most similar reference object
    iso_knn     1 - mean similarity to the min_pts most similar
    iso_degree  1 / (1 + how many reference objects exceed lsh.min_sim)

They are here because a per-column score is blind to the fault that matters
most: a combination of individually ordinary values. High irradiance with low
output is anomalous as a pair while neither column is unusual alone, and no sum
over per-column deviations can see it. Measured on the same days, a plain
distance to the nearest training object put 11 true anomalies in the top 100
where the best per-column score managed 5, and it detected all four injected
fault kinds instead of mostly spikes.

On unit vectors these are the same quantity a Euclidean distance measures --
||a-b||^2 = 2(1 - cos(a,b)) -- so `iso_nn` ranks exactly as a 1-NN distance
does. What DENCAST adds over that is `iso_degree`: counting the neighbourhood
rather than trusting its nearest member, which is the `min_pts` criterion
applied to an unseen object and something a centroid-based method cannot express.

A sigmoid is not applied to any of these: it is monotonic, so it cannot change
their ranking, and the bounded form is only needed where a fixed threshold or a
product with another term is wanted.
"""

DEFAULT_BATCH_DAYS = 10
"""Scored days per Spark job.

This is a memory bound, and it was measured rather than guessed. Each query
object draws `2 * b * num_permutations` candidates -- 1,800 at b=30 and 30
permutations, because the routing window runs both ways -- and every surviving
candidate carries two 17-component arrays into the cosine. A batch of 45 days is
some eleven thousand objects, hence twenty million wide rows, which exhausted a
three-gigabyte heap. Ten days is roughly 2,500 objects.

The days are independent once the model is fitted, so this changes no result.
"""


def apply_overrides(params: Params, overrides: Dict[str, Any]) -> Params:
    """A copy of `params` with `section.field` overrides applied.

    Deep-copied rather than mutated: a grid point that quietly edited the shared
    object would make every later point inherit it, and the bug would look like a
    hyperparameter effect.
    """
    out = copy.deepcopy(params)
    for path, value in overrides.items():
        section, _, field_name = path.partition(".")
        setattr(getattr(out, section), field_name, value)
    out.validate()
    return out


def deviation_columns(params: Params) -> List[str]:
    """Columns entering the deviation: everything in use, less the excluded."""
    excluded = set(params.scoring.exclude_cols)
    return [c for c in params.dataset.feature_cols if c not in excluded]


def on_days(df: DataFrame, days: Sequence[str]) -> DataFrame:
    """The rows falling on any of `days`."""
    return df.filter(F.col("date").isin(list(days)))


def fit_split(
    spark: SparkSession, df: DataFrame, params: Params, days: Sequence[str], seed: int
) -> DencastModel:
    """Fit one model on every row of the given days."""
    train = on_days(df, days).repartition(params.spark.num_partitions).cache()
    n = train.count()
    if n == 0:
        raise ValueError("Lo split di train non contiene righe")
    logger.info("Fit su {} giorni, {} oggetti", len(days), f"{n:,}")
    fitted = fit(spark, train, params, seed)
    logger.info("  {}", fitted.summary())
    return fitted


def score_days(
    spark: SparkSession,
    model: DataFrame,
    values: DataFrame,
    params: Params,
    days: Sequence[str],
    dates: DataFrame,
    seed: int,
    label_col: str = "anomaly",
    batch_days: int = DEFAULT_BATCH_DAYS,
    include_noise: bool = False,
) -> Tuple[Dict[str, List[float]], List[int], List[int], Dict[str, float]]:
    """Score every object on `days` against the already-fitted model.

    Args:
        model: (id, cluster_id, ...) from a fitted DencastModel.
        values: (id, <columns>, <label>) in min-max scale. Not the parquet's
            `features`, which is L2-normalised and so couples the columns: a
            per-column deviation computed on it would measure the norm.
        dates: (id, date), to pick out the days being scored.

    Returns (scores per formula, labels, ids, diagnostics), all index-aligned.
    Labels are read only here, after every score is already fixed.
    """
    from dencast.anomaly_metrics import evaluate_ranking

    routing_cols = list(params.dataset.feature_cols)
    dev_cols = deviation_columns(params)

    labeled = (
        model.select("id", "cluster_id").join(values, on="id", how="inner").cache()
    )
    scores: Dict[str, List[float]] = {f: [] for f in FORMULAS}
    labels: List[int] = []
    ids: List[int] = []
    sims: List[float] = []
    degrees: List[int] = []
    n_unrouted = 0
    index = None
    try:
        # Both of these depend only on the model, so they are built once and
        # reused by every batch. Hashing the reference set is the expensive half
        # of the routing; recomputing it per batch was pure waste.
        moments = column_moments(
            labeled.select("id", "cluster_id"), values, dev_cols, params.scoring.shrinkage
        ).cache()
        moments.count()
        quantiles = column_quantiles(
            labeled.select("id", "cluster_id"), values, dev_cols
        ).cache()
        quantiles.count()
        index = build_routing_index(
            labeled, routing_cols, params, seed, include_noise=include_noise
        )

        batches = [days[i : i + batch_days] for i in range(0, len(days), batch_days)]
        for bi, batch in enumerate(batches, start=1):
            target_ids = dates.filter(F.col("date").isin(list(batch))).select("id").cache()
            try:
                n_target = target_ids.count()
                unlabeled = target_ids.join(values, on="id", how="inner")
                routed = route_batch(unlabeled, index, routing_cols, params)
                # The routing is the expensive part of this job and three
                # things read it, so it is materialised before they branch.
                routed = routed.localCheckpoint(eager=True)
                z_part = sigmoid_scores(
                    deviation_scores(routed, values, moments, dev_cols),
                    params.scoring.offset,
                    params.scoring.scale,
                )
                q_part = quantile_scores(routed, values, quantiles, dev_cols).select(
                    "id", "q_max", "q_top3", "q_sum"
                )
                # Isolation in the joint space, straight off the routing: no
                # per-column step, so nothing here can be blind to a combination
                # of individually ordinary values.
                iso_part = routed.select(
                    "id",
                    (F.lit(1.0) - F.col("best_sim")).alias("iso_nn"),
                    (F.lit(1.0) - F.col("top_mean_sim")).alias("iso_knn"),
                    (F.lit(1.0) / (F.lit(1.0) + F.col("degree"))).alias("iso_degree"),
                    "degree",
                    "n_candidates",
                )
                scored = (
                    z_part.join(q_part, on="id", how="inner")
                    .join(iso_part, on="id", how="inner")
                    .localCheckpoint(eager=True)
                )
                rows = (
                    scored.join(values.select("id", label_col), on="id", how="inner")
                    .select("id", *FORMULAS, "best_sim", "degree", label_col)
                    .collect()
                )
                for f in FORMULAS:
                    scores[f].extend(float(r[f]) for r in rows)
                sims.extend(float(r["best_sim"]) for r in rows)
                degrees.extend(int(r["degree"]) for r in rows)
                labels.extend(1 if r[label_col] else 0 for r in rows)
                ids.extend(int(r["id"]) for r in rows)

                # Objects LSH found no candidate for. Scored at zero rather than
                # dropped: dropping them would quietly shrink the denominator and
                # turn a miss into a non-event, which flatters every metric.
                # Scored at zero they count as "nothing found", which is what
                # actually happened.
                #
                # A left-anti join, not `isin` over the collected ids: that list
                # runs to thousands, and every element becomes a query-plan
                # literal.
                missing = (
                    target_ids.join(scored.select("id"), on="id", how="left_anti")
                    .join(values.select("id", label_col), on="id", how="inner")
                    .select("id", label_col)
                    .collect()
                )
                for r in missing:
                    for f in FORMULAS:
                        scores[f].append(0.0)
                    sims.append(0.0)
                    labels.append(1 if r[label_col] else 0)
                    ids.append(int(r["id"]))
                n_unrouted += len(missing)

                logger.info(
                    "  [{}/{}] {} giorni  {} oggetti ({} anomali, {} non instradati)   "
                    "AUC-PR sigmoid_plain {:.4f}",
                    bi, len(batches), len(batch), n_target,
                    sum(1 for r in rows if r[label_col]), len(missing),
                    evaluate_ranking(scores["sigmoid_plain"], labels)["average_precision"],
                )
            finally:
                target_ids.unpersist()
        moments.unpersist()
        quantiles.unpersist()
    finally:
        if index is not None:
            index.unpersist()
        labeled.unpersist()

    if not labels:
        raise ValueError("Nessun oggetto valutato in questo split")

    # Saturation, measured rather than assumed. `sigmoid_plain` is a monotonic
    # function of `sum_sq`, so it must produce the same number of distinct
    # values; fewer means float64 tied scores that the raw quantity separates,
    # and its ranking is degraded by exactly that much.
    diagnostics = {
        "n_scored": float(len(labels)),
        "n_anomalies": float(sum(labels)),
        "n_unrouted": float(n_unrouted),
        "frac_unrouted": float(n_unrouted / len(labels)),
        "mean_best_sim": float(sum(sims) / len(sims)),
        # The neighbourhood sizes the density scores rest on. A mean degree far
        # above min_pts means almost nothing looks isolated and iso_degree has
        # little to separate; far below, and the threshold is too strict for the
        # reference set to support it.
        "include_noise": float(include_noise),
        "mean_degree": float(sum(degrees) / len(degrees)) if degrees else 0.0,
        "frac_below_min_pts": (
            float(sum(1 for g in degrees if g < params.clustering.min_pts) / len(degrees))
            if degrees else 0.0
        ),
        "distinct_sigmoid_plain": float(len(set(scores["sigmoid_plain"]))),
        "distinct_sum_sq": float(len(set(scores["sum_sq"]))),
        "frac_sigmoid_at_one": float(
            sum(1 for v in scores["sigmoid_plain"] if v >= 1.0) / len(labels)
        ),
    }
    return scores, labels, ids, diagnostics


def evaluate_all(
    scores: Dict[str, List[float]], labels: List[int]
) -> Dict[str, Dict[str, float]]:
    """Ranking metrics for every formula."""
    from dencast.anomaly_metrics import evaluate_ranking

    return {f: evaluate_ranking(v, labels) for f, v in scores.items()}


def build_values(spark: SparkSession, params: Params) -> DataFrame:
    """The min-max column values and the label, read from the source CSV.

    Read from the source rather than the parquet on purpose: the parquet's
    `features` array is L2-normalised, which ties the columns to each other, so a
    per-column z computed on it would be measuring the norm. The source of these
    datasets is already min-max scaled, which is what the deviation wants.
    """
    cols = list(params.dataset.feature_cols)
    label = params.dataset.anomaly_col
    wanted = ["id", *cols] + ([label] if label else [])
    return (
        spark.read.option("header", "true").option("inferSchema", "true")
        .csv(params.dataset.source)
        .select(*wanted)
    )


__all__ = [
    "FORMULAS",
    "DEFAULT_BATCH_DAYS",
    "apply_overrides",
    "deviation_columns",
    "on_days",
    "fit_split",
    "score_days",
    "evaluate_all",
    "build_values",
]
