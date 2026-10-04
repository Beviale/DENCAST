"""The DENCAST model: building it, and describing the clustering it found.

    LSH  ->  neighborhood graph  ->  core objects  ->  propagation  ->  clusters

This is the library, not a pipeline stage. DENCAST is instance-based, so there
are no learned weights: the model *is* the set of labeled objects plus their
cluster assignment. `train_and_select` calls `fit` once per grid point per day,
and `train_winner` calls it once on the whole train split.

The training phase never sees the unlabeled objects and the prediction phase
never modifies the graph, which is what makes a fitted model reusable across
the days it is asked to predict.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from dencast.clustering import NOISE_CLUSTER_ID, cluster_summary, find_clusters
from dencast.lsh import build_neighborhood_graph
from dencast.utils import Params


@dataclass
class DencastModel:
    """A fitted model: the labeled objects plus their cluster assignment."""

    model: DataFrame
    """(id, features, targets, cluster_id) for every labeled object."""

    edges: DataFrame
    """(src, dst, sim), the neighborhood graph. Kept because anything built on
    top of DENCAST -- degrees, local density, the attachment strength of a new
    object -- needs the graph and not just the cluster labels."""

    stats: dict

    def summary(self) -> str:
        s = self.stats
        return (
            f"clusters={s['n_clusters']}  core={s['n_core']}  noise={s['n_noise']}  "
            f"edges={s['n_edges']}  iterations={s['n_iterations']}  "
            f"{s['fit_seconds']:.1f}s"
        )


def fit(spark: SparkSession, labeled: DataFrame, params: Params, seed: int) -> DencastModel:
    """Build the DENCAST model from the labeled objects.

    Args:
        labeled: (id, features, targets).
        params: project parameters.
        seed: seed for the LSH hyperplanes and permutations. Different seeds
            give different graphs and therefore different clusters, so this is
            part of the experiment, not an implementation detail.
    """
    params.validate()
    start = time.time()

    labeled = labeled.select("id", "features", "targets").cache()
    n_labeled = labeled.count()
    logger.info("Training objects: {}", n_labeled)

    # Step 1: approximate neighborhood graph via LSH.
    vertices, edges = build_neighborhood_graph(spark, labeled, params, seed)
    edges = edges.cache()
    n_edges = edges.count()

    # Steps 2 and 3: core objects, then propagation of the cluster IDs.
    clusters, n_iterations = find_clusters(vertices, edges, params)
    clusters = clusters.cache()

    n_core = clusters.filter(F.col("is_core")).count()
    n_noise = clusters.filter(F.col("cluster_id") == F.lit(NOISE_CLUSTER_ID)).count()
    n_clusters = (
        clusters.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID))
        .select("cluster_id")
        .distinct()
        .count()
    )

    # Step 5 of Algorithm 1: bring the original features back alongside the
    # cluster assignment. During the clustering the graph carried only ids,
    # which is what keeps the iterative shuffles small.
    model = labeled.join(clusters.select("id", "cluster_id"), on="id", how="left").select(
        "id",
        "features",
        "targets",
        F.coalesce(F.col("cluster_id"), F.lit(NOISE_CLUSTER_ID)).alias("cluster_id"),
    )

    stats = {
        "n_labeled": n_labeled,
        "n_edges": n_edges,
        "n_clusters": n_clusters,
        "n_core": n_core,
        "n_noise": n_noise,
        "noise_fraction": n_noise / n_labeled if n_labeled else 0.0,
        "n_iterations": n_iterations,
        "fit_seconds": time.time() - start,
        "seed": seed,
    }
    fitted = DencastModel(model=model.cache(), edges=edges, stats=stats)
    logger.info("Fitted: {}", fitted.summary())
    return fitted


def cluster_diagnostics(fitted: DencastModel) -> dict:
    """Structural diagnostics of the clustering.

    The cluster count turns out to be a better tuning guide than RMSE: it says
    whether the structure being recovered is the right one, and it is far less
    sensitive to which particular test day was picked. Table 4 of the paper
    reports 618 clusters for PV Italy ST at 30 days.
    """
    counts = [row["size"] for row in cluster_summary(fitted.model).collect()]
    if not counts:
        return {
            "cluster_size_max": 0,
            "cluster_size_mean": 0.0,
            "cluster_size_median": 0,
            "singleton_clusters": 0,
        }
    ordered = sorted(counts)
    return {
        "cluster_size_max": max(counts),
        "cluster_size_mean": sum(counts) / len(counts),
        "cluster_size_median": ordered[len(ordered) // 2],
        "singleton_clusters": sum(1 for c in counts if c == 1),
    }
