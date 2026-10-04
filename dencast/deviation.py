"""Scoring an object against its cluster, one column at a time.

A different formulation from `dencast.anomaly`. There, one column is the
target, the cluster predicts it, and the residual is the signal -- which makes
an anomaly in any *other* column invisible: it does not produce an error, it
just routes the object somewhere else, where it looks unremarkable.

Here there is no target. Every column is compared against the cluster the
object was routed to:

    z_j  =  | x_j - mu_j(cluster) |  /  sigma_j(cluster)

and the column scores are combined. Nothing predicts anything, so Eq. 2 and
the weighted average drop out entirely and the only thing the clustering is
asked for is a reference population.

**Aggregation is a choice with a measurable cost.** Taking the maximum asks
"is it out of place in at least one column"; summing the squares asks "how far
out of place overall". On simulated anomalies with sixteen columns the
difference is not symmetric: for a single column shifted by 5 sigma the
maximum separates at AUC 0.996 against the sum's 0.982, but for eight columns
shifted by 1.5 sigma each the sum reaches 0.948 against the maximum's 0.879.
A quiet fault that spreads across sensors is exactly what the maximum misses,
so the sum is the default here, and `top_k` interpolates: the root of the sum
of the k largest squares, which is the maximum at k=1 and the full sum at
k=n_columns.

**The values must not be L2-normalised.** Our processed features are, because
that is what the cosine similarity of the routing wants. But L2 normalisation
couples the columns -- change one and every other changes with it -- so a
per-column deviation computed on them would measure an artefact. Pass the
min-max scaled values instead; the routing can go on using the normalised
ones.

**The variance is shrunk, as elsewhere.** A cluster of six members gives a
per-column variance estimated from six numbers, and dividing by a spuriously
small one manufactures anomalies. The same empirical-Bayes pull toward the
global variance is applied, with the same pseudo-count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from dencast.anomaly import DEFAULT_SHRINKAGE
from dencast.clustering import NOISE_CLUSTER_ID

DEFAULT_KS = (1, 3, 5)
"""Aggregation widths reported alongside the full sum, so the choice between
"out of place in one column" and "out of place overall" can be read off the
same run instead of being fixed in advance."""


def column_moments(
    model: DataFrame, values: DataFrame, columns: Sequence[str], shrinkage: float
) -> DataFrame:
    """Per cluster and column: the mean and a shrunk standard deviation.

    Returns (cluster_id, col, mu, sigma, n). Built from the sufficient
    statistics in one pass -- count, sum, sum of squares -- so the cost does
    not grow with how many objects are later scored against them.
    """
    clustered = model.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID))
    joined = clustered.select("id", "cluster_id").join(values, on="id", how="inner")

    long = joined.select(
        "cluster_id",
        F.explode(
            F.array(*[F.struct(F.lit(c).alias("col"), F.col(c).cast("double").alias("v"))
                      for c in columns])
        ).alias("cv"),
    ).select("cluster_id", F.col("cv.col").alias("col"), F.col("cv.v").alias("v"))

    per_cluster = long.groupBy("cluster_id", "col").agg(
        F.count("*").alias("n"),
        F.sum("v").alias("s1"),
        F.sum(F.col("v") * F.col("v")).alias("s2"),
    )
    # The prior each cluster is pulled toward: the spread of that column over
    # every clustered object, which is what a cluster of two has no view of.
    per_column = long.groupBy("col").agg(
        (
            (F.sum(F.col("v") * F.col("v")) - F.sum("v") * F.sum("v") / F.count("*"))
            / F.greatest(F.count("*") - F.lit(1), F.lit(1))
        ).alias("var_global")
    )

    return (
        per_cluster.join(per_column, on="col", how="left")
        .withColumn("mu", F.col("s1") / F.col("n"))
        .withColumn(
            "var_own",
            F.when(
                F.col("n") > 1,
                F.greatest(
                    (F.col("s2") - F.col("s1") * F.col("s1") / F.col("n"))
                    / (F.col("n") - F.lit(1)),
                    F.lit(0.0),
                ),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "sigma",
            F.sqrt(
                (F.col("n") * F.col("var_own") + F.lit(shrinkage) * F.coalesce(F.col("var_global"), F.lit(0.0)))
                / (F.col("n") + F.lit(shrinkage))
            ),
        )
        .select("cluster_id", "col", "mu", "sigma", "n")
    )


def deviation_scores(
    routed: DataFrame,
    values: DataFrame,
    moments: DataFrame,
    columns: Sequence[str],
    ks: Sequence[int] = DEFAULT_KS,
    min_sigma: float = 1e-9,
) -> DataFrame:
    """How far each object sits from its cluster, column by column.

    Args:
        routed: (id, cluster_id, best_sim) -- the output of `predict`, which
            supplies the anchor and how similar it was.
        values: (id, <columns>) in min-max scale, not L2-normalised.
        moments: the output of `column_moments`.
        columns: the columns to score.
        ks: widths for the top-k aggregation.

    Returns one row per object: the per-column deviations aggregated several
    ways, plus `best_sim` untouched so the two signals -- "it does not fit its
    cluster" and "nothing in training looked like it" -- can be weighed
    separately rather than blended here.
    """
    long = routed.select("id", "cluster_id", "best_sim").join(
        values.select("id", *columns), on="id", how="inner"
    ).select(
        "id", "cluster_id", "best_sim",
        F.explode(
            F.array(*[F.struct(F.lit(c).alias("col"), F.col(c).cast("double").alias("v"))
                      for c in columns])
        ).alias("cv"),
    ).select("id", "cluster_id", "best_sim", F.col("cv.col").alias("col"), F.col("cv.v").alias("v"))

    with_z = (
        long.join(moments, on=["cluster_id", "col"], how="left")
        .withColumn("sigma_eff", F.greatest(F.coalesce(F.col("sigma"), F.lit(min_sigma)), F.lit(min_sigma)))
        .withColumn("z", F.abs(F.col("v") - F.col("mu")) / F.col("sigma_eff"))
    )

    # Squares gathered per object and sorted descending once; every
    # aggregation is then a prefix of the same array.
    gathered = with_z.groupBy("id", "cluster_id", "best_sim").agg(
        F.reverse(F.array_sort(F.collect_list(F.col("z") * F.col("z")))).alias("z2"),
        F.max("z").alias("z_max"),
        F.avg("z").alias("z_mean"),
        F.count("*").alias("n_columns"),
    )

    def rss(arr):
        return F.sqrt(F.aggregate(arr, F.lit(0.0), lambda acc, x: acc + x))

    out = gathered.withColumn(
        # The sum of squares itself, not its root: this is the S the sigmoid
        # formulas take, and rooting it first then squaring it back would only
        # lose precision.
        "sum_sq", F.aggregate(F.col("z2"), F.lit(0.0), lambda acc, x: acc + x)
    ).withColumn("score_sum", F.sqrt(F.col("sum_sq")))
    for k in ks:
        out = out.withColumn(f"score_top{k}", rss(F.slice(F.col("z2"), 1, k)))

    return out.drop("z2")


def score_objects(
    model: DataFrame,
    routed: DataFrame,
    values: DataFrame,
    columns: Sequence[str],
    shrinkage: float = DEFAULT_SHRINKAGE,
    ks: Sequence[int] = DEFAULT_KS,
) -> DataFrame:
    """Convenience: moments from the model, then deviations for the routed objects."""
    moments = column_moments(model, values, columns, shrinkage)
    return deviation_scores(routed, values, moments, columns, ks)


def unit_vector(columns: Sequence[str]):
    """The row's columns as one L2-normalised vector, for the cosine.

    Built from the min-max values rather than reused from the processed
    parquet, whose `features` array is L2-normalised over the descriptive
    attributes *only* -- the target was never part of that norm.
    """
    raw = F.array(*[F.coalesce(F.col(c).cast("double"), F.lit(0.0)) for c in columns])
    length = F.sqrt(F.aggregate(raw, F.lit(0.0), lambda acc, x: acc + x * x))
    return F.when(
        length > 0, F.transform(raw, lambda x: x / length)
    ).otherwise(raw)


def route(
    unlabeled: DataFrame, labeled: DataFrame, columns: Sequence[str]
) -> DataFrame:
    """Eq. 1, but with nothing held back: the anchor is found on all columns.

    `modeling.predict.route`-equivalent, except that the similarity there runs
    over the descriptive attributes only, because the target is exactly what
    it is about to predict. Here nothing is being predicted, so there is no
    reason to hide a column -- and a positive one to include it. A corrupted
    value drags the object away from everything in the training window, which
    shows up as a low `best_sim`. Route on the features alone and that
    evidence is discarded before the score is ever computed.

    Returns (id, cluster_id, best_sim). Noise objects are not eligible as
    anchors: they belong to no cluster, so there would be no population to
    measure a deviation against.
    """
    from dencast.spark_session import cosine_similarity

    candidates = labeled.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID)).select(
        unit_vector(columns).alias("l_vec"), F.col("cluster_id")
    )
    u = unlabeled.select(F.col("id").alias("u_id"), unit_vector(columns).alias("u_vec"))

    sims = u.crossJoin(candidates).withColumn(
        "sim", cosine_similarity(F.col("u_vec"), F.col("l_vec"))
    )
    # max over a struct compares field by field, so ties on the similarity
    # fall back to the cluster id and the choice stays reproducible.
    return (
        sims.groupBy("u_id")
        .agg(F.max(F.struct("sim", "cluster_id")).alias("best"))
        .select(
            F.col("u_id").alias("id"),
            F.col("best.cluster_id").alias("cluster_id"),
            F.col("best.sim").alias("best_sim"),
        )
    )


DEFAULT_OFFSET = 1.0
"""Where the sigmoid crosses a half, in mean z-squared per column.

A well-behaved object sits about one standard deviation from its cluster mean
in a typical column, so its mean z-squared is about 1. Putting the midpoint
there makes the score read as "more out of place than an ordinary member".
"""

DEFAULT_SCALE = 1.0
"""How many units of mean z-squared the sigmoid takes to go from 0.12 to 0.88."""


def sigmoid_scores(
    scored: DataFrame,
    offset: float = DEFAULT_OFFSET,
    scale: float = DEFAULT_SCALE,
) -> DataFrame:
    """The two bounded scores, from `sum_sq` and `best_sim`.

        S      = sum over columns of z_j squared
        mean   = S / n_columns
        plain  = sigmoid((mean - offset) / scale)
        scaled = sigmoid((mean * best_sim - offset) / scale)

    **The sigmoid cannot improve a ranking on its own.** It is monotonic, so
    `sigmoid(S)` orders objects exactly as `S` does and every ranking metric --
    AUC-PR, ROC-AUC, precision@k -- comes out identical. What it buys is a
    bounded number with a fixed operating point, which is what makes a
    threshold meaningful and what lets the second formula multiply by a
    similarity without one term swamping the other.

    **Which is also why the offset and the scale are not cosmetic.** S runs to
    the tens or hundreds across seventeen columns, and float64 cannot tell
    `sigmoid(36)` from `sigmoid(200)`: both are exactly 1.0. Feed S in raw and
    every object ties at the ceiling, which reads as a detector that has failed
    when in fact only the transform has. Dividing by the column count and
    centring at `offset` keeps the argument in the range where the sigmoid
    still resolves differences.

    `sum_sq` is carried through unchanged so the un-saturated ranking stays
    available: whatever the sigmoid variants score, the raw S is what they
    would score with a perfectly chosen scale.
    """
    mean_z2 = F.col("sum_sq") / F.greatest(F.col("n_columns"), F.lit(1))
    plain = (mean_z2 - F.lit(offset)) / F.lit(scale)
    # Multiplying before centring, as specified: the similarity damps the
    # deviation itself rather than shifting the operating point. A confidently
    # routed object that still deviates keeps its score; one whose nearest
    # neighbour was distant has its deviation discounted.
    scaled = (mean_z2 * F.col("best_sim") - F.lit(offset)) / F.lit(scale)
    return (
        scored.withColumn("mean_z2", mean_z2)
        .withColumn("sigmoid_plain", F.lit(1.0) / (F.lit(1.0) + F.exp(-plain)))
        .withColumn("sigmoid_sim", F.lit(1.0) / (F.lit(1.0) + F.exp(-scaled)))
        .withColumn("raw_sim", mean_z2 * F.col("best_sim"))
    )


@dataclass
class RoutingIndex:
    """The reference side of the routing, prepared once.

    Hashing 162,000 objects is the expensive half of the routing, and it does
    not depend on which batch is being scored. Building it once and reusing it
    is the difference between doing that work once and doing it per batch.

    `bits` and `vectors` must come from the same `columns` and the same `seed`,
    which is why they are produced together rather than by the caller.
    """

    vectors: DataFrame
    """(id, cluster_id, lsh_vec) -- unit vectors over all routing columns."""

    bits: DataFrame
    """(id, bits) -- the r-bit signatures."""

    n_features: int
    seed: int

    def unpersist(self) -> None:
        self.vectors.unpersist()
        self.bits.unpersist()


def build_routing_index(
    labeled: DataFrame,
    columns: Sequence[str],
    params,
    seed: int,
    include_noise: bool = False,
) -> RoutingIndex:
    """Prepare the reference side: unit vectors and signatures, cached.

    `include_noise` decides whether the objects DENCAST labelled noise may serve
    as neighbours. It is a modelling choice with arguments both ways, and the
    right default depends on what the score is for.

    For the per-column deviation, excluding them is forced: a noise object
    belongs to no cluster, so it has no mean or variance to be compared against.

    For a density or nearest-neighbour score it is a real decision. The training
    window is contaminated on purpose -- anomalies were injected across the whole
    dataset -- so if an anomaly's nearest neighbour is another anomaly, keeping
    noise in makes it look ordinary: something similar exists. Dropping noise
    moves the reference closer to "normal behaviour only", which is unsupervised
    cleaning DENCAST supplies for free. Against that, some of those objects are
    normal but rare -- an under-represented plant, an unusual day -- and removing
    them leaves those regions artificially empty, so every legitimate reading
    falling there looks isolated. Which effect wins is measured, not argued: the
    exact all-objects baseline (sklearn, noise included) scored 11 true anomalies
    in the top 100, and comparing it against an LSH index that also drops noise
    would confound the approximation with the filtering.
    """
    from dencast.lsh import compute_signatures

    eligible = (
        labeled
        if include_noise
        else labeled.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID))
    )
    vectors = eligible.select(
        "id", "cluster_id", unit_vector(columns).alias("lsh_vec")
    ).cache()
    vectors.count()
    bits = (
        compute_signatures(vectors, "lsh_vec", len(columns), params.lsh.r, seed)
        .select("id", "bits")
        .cache()
    )
    bits.count()
    return RoutingIndex(vectors=vectors, bits=bits, n_features=len(columns), seed=seed)


def route_batch(
    unlabeled: DataFrame,
    index: RoutingIndex,
    columns: Sequence[str],
    params,
) -> DataFrame:
    """Eq. 1 for one batch, affordably, without giving up the cluster shapes.

    The exact form compares every object to be scored against every reference
    object: 43,000 x 162,000 is seven billion cosines here. The cheap
    alternative is to compare against one mean vector per cluster, which is what
    the paper's Scala code does -- and which is wrong for this algorithm. A mean
    represents a compact blob; a density-based cluster is free to be long or
    curved, and its mean can fall outside it entirely, nearer another cluster's
    members than its own. Routing by centroid would quietly turn inference into
    K-means and discard the property that motivated DENCAST.

    So this blocks with LSH, exactly as the graph does: same hyperplanes, same
    permutations, same sliding window. Each object gets `2 * b *
    num_permutations` candidates -- a constant, independent of the reference
    size -- and the cosine is computed exactly on those. Real nodes, so
    arbitrary shapes survive; and no new kind of error, since this is the
    approximation the graph already accepts.

    **Batch size is a memory decision, not a correctness one.** The candidates
    carry two 17-component arrays into the cosine, so a batch of eleven thousand
    objects at 1,800 candidates each is twenty million wide rows, which does not
    fit in a three-gigabyte heap. The days are independent once the model is
    fitted, so batching changes no result.

    Returns (id, cluster_id, best_sim) for the objects that got at least one
    candidate. Objects with none are absent: that is this method's recall loss,
    and the caller is expected to count them rather than let them vanish.
    """
    from dencast.lsh import compute_signatures, cross_candidate_pairs_from_bits
    from dencast.spark_session import cosine_similarity

    q_vec = unlabeled.select("id", unit_vector(columns).alias("lsh_vec"))
    q_bits = compute_signatures(
        q_vec, "lsh_vec", index.n_features, params.lsh.r, index.seed
    ).select("id", "bits")

    candidates = cross_candidate_pairs_from_bits(q_bits, index.bits, params, index.seed)
    sims = (
        candidates.join(
            q_vec.select(F.col("id").alias("q_id"), F.col("lsh_vec").alias("q_vec")),
            on="q_id",
            how="inner",
        )
        .join(
            index.vectors.select(
                F.col("id").alias("r_id"),
                F.col("lsh_vec").alias("r_vec"),
                "cluster_id",
            ),
            on="r_id",
            how="inner",
        )
        .withColumn("sim", cosine_similarity(F.col("q_vec"), F.col("r_vec")))
    )
    # Everything the candidate similarities can say, aggregated in one pass.
    # The routing computed all of them and then kept only the maximum; the rest
    # were the local density, discarded. Density is what a density-based
    # clustering is for, and on unit vectors it is the same quantity a
    # nearest-neighbour distance measures: ||a-b||^2 = 2(1 - cos(a,b)), so a
    # distance score and a similarity score rank identically. The winning
    # baseline on this data was exactly that, computed outside DENCAST.
    #
    # max over a struct compares field by field, so a tie on the similarity
    # falls back to the cluster id and the choice stays reproducible.
    top_pts = max(int(params.clustering.min_pts), 1)
    return (
        sims.groupBy("q_id")
        .agg(
            F.max(F.struct("sim", "cluster_id")).alias("best"),
            # How many reference objects are close enough to count as
            # neighbours, by the same threshold the graph is built with. This is
            # the min_pts criterion applied to an unseen object, and it is more
            # robust than the single nearest similarity: one lucky neighbour
            # rescues an isolated object from a 1-NN score but not from a count.
            F.sum(
                F.when(F.col("sim") >= F.lit(params.lsh.min_sim), F.lit(1)).otherwise(F.lit(0))
            ).alias("degree"),
            F.reverse(F.array_sort(F.collect_list("sim"))).alias("sims_desc"),
            F.count("*").alias("n_candidates"),
        )
        .select(
            F.col("q_id").alias("id"),
            F.col("best.cluster_id").alias("cluster_id"),
            F.col("best.sim").alias("best_sim"),
            "degree",
            "n_candidates",
            # Mean of the min_pts most similar, i.e. a k-NN score rather than a
            # 1-NN one. Averaging over the neighbourhood is what makes it less
            # sensitive to a single coincidental match.
            F.aggregate(
                F.slice(F.col("sims_desc"), 1, top_pts), F.lit(0.0), lambda acc, x: acc + x
            ).alias("_sum_top"),
            F.size(F.slice(F.col("sims_desc"), 1, top_pts)).alias("_n_top"),
        )
        .withColumn(
            "top_mean_sim",
            F.when(F.col("_n_top") > 0, F.col("_sum_top") / F.col("_n_top")).otherwise(
                F.lit(0.0)
            ),
        )
        .drop("_sum_top", "_n_top")
    )


def cluster_centroids(labeled: DataFrame, columns: Sequence[str]) -> DataFrame:
    """One mean vector per cluster, L2-normalised, for the cosine.

    Returns (cluster_id, c_vec, n_members). The mean is taken on the min-max
    values and normalised afterwards, which is the order that matters: the mean
    of unit vectors is not the unit vector of the mean, and the cosine wants the
    latter.
    """
    long = labeled.select(
        "cluster_id",
        F.posexplode(
            F.array(*[F.coalesce(F.col(c).cast("double"), F.lit(0.0)) for c in columns])
        ).alias("pos", "v"),
    )
    means = (
        long.groupBy("cluster_id", "pos")
        .agg(F.avg("v").alias("m"))
        .groupBy("cluster_id")
        .agg(F.sort_array(F.collect_list(F.struct("pos", "m"))).alias("pairs"))
        .select(
            "cluster_id",
            F.transform(F.col("pairs"), lambda p: p["m"]).alias("raw"),
        )
    )
    length = F.sqrt(F.aggregate(F.col("raw"), F.lit(0.0), lambda acc, x: acc + x * x))
    sizes = labeled.groupBy("cluster_id").count().withColumnRenamed("count", "n_members")
    return (
        means.withColumn(
            "c_vec",
            F.when(length > 0, F.transform(F.col("raw"), lambda x: x / length)).otherwise(
                F.col("raw")
            ),
        )
        .drop("raw")
        .join(sizes, on="cluster_id", how="inner")
    )


def route_via_centroids(
    unlabeled: DataFrame,
    labeled: DataFrame,
    columns: Sequence[str],
    refine: bool = True,
) -> DataFrame:
    """Eq. 1 made affordable: pick the cluster by centroid, then the node inside it.

    Comparing every unlabeled object against every labeled one is exact and
    quadratic: on this dataset it is 43,000 x 162,000 = seven billion cosines,
    which is tens of hours. This is two cheap stages instead.

    Stage one compares against one mean vector per cluster -- a few hundred,
    not a hundred thousand -- which is what the paper's own Scala
    implementation does (`predict` notes the discrepancy: the code routes by
    centroid where the paper routes by nearest object).

    Stage two, when `refine` is on, recomputes the exact cosine against the
    members of the chosen cluster only, so `best_sim` is the similarity to a
    real nearest node rather than to an average of many. That costs the size of
    one cluster per object instead of the whole training set.

    **The approximation is named, not hidden.** The truly most similar node may
    sit in a cluster whose centroid was not the closest -- a long or curved
    cluster is exactly where a mean vector misrepresents its members. What comes
    back is the best node within the best-centroid cluster, which is the same
    compromise the original implementation ships.

    Returns (id, cluster_id, best_sim).
    """
    from dencast.spark_session import cosine_similarity

    members = labeled.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID))
    centroids = cluster_centroids(members, columns).select("cluster_id", "c_vec")
    u = unlabeled.select(F.col("id").alias("u_id"), unit_vector(columns).alias("u_vec"))

    # Stage 1: the cluster. Ties fall back to the cluster id so the choice is
    # reproducible rather than decided by partition order.
    chosen = (
        u.crossJoin(centroids)
        .withColumn("sim", cosine_similarity(F.col("u_vec"), F.col("c_vec")))
        .groupBy("u_id")
        .agg(F.max(F.struct("sim", "cluster_id")).alias("best"))
        .select(
            "u_id",
            F.col("best.cluster_id").alias("cluster_id"),
            F.col("best.sim").alias("centroid_sim"),
        )
    )
    if not refine:
        return chosen.select(
            F.col("u_id").alias("id"), "cluster_id",
            F.col("centroid_sim").alias("best_sim"),
        )

    # Stage 2: the nearest actual member of that cluster.
    member_vecs = members.select(
        F.col("cluster_id").alias("m_cluster"), unit_vector(columns).alias("m_vec")
    )
    refined = (
        u.join(chosen, on="u_id", how="inner")
        .join(member_vecs, F.col("cluster_id") == F.col("m_cluster"), how="inner")
        .withColumn("sim", cosine_similarity(F.col("u_vec"), F.col("m_vec")))
        .groupBy("u_id", "cluster_id")
        .agg(F.max("sim").alias("best_sim"))
    )
    return refined.select(F.col("u_id").alias("id"), "cluster_id", "best_sim")


def novelty_column(best_sim, eps: float = 1e-9):
    """The other signal: how unlike anything in training the object was.

    `-log(1 - sim)` is large when the nearest neighbour is close, so the
    novelty is its negation -- an object whose best match is distant scores
    high. Kept apart from the deviation on purpose: the calibration curve `g`
    in `dencast.anomaly` divides the score *down* when the anchor is far,
    treating distance as an excuse, and this treats it as evidence. They
    cannot both be right, and which one is depends on the data, so neither is
    baked in here.
    """
    return -F.log(F.greatest(F.lit(1.0) - best_sim, F.lit(eps)))


def combine(
    scored: DataFrame,
    deviation_col: str = "score_sum",
    weight: float = 1.0,
    out_col: str = "score_combined",
) -> DataFrame:
    """Deviation and novelty in one number, with the weight exposed.

    novelty enters as `(1 - best_sim)` scaled: an object that fits its cluster
    but resembles nothing gets lifted, one that fits and had a near-identical
    neighbour does not.
    """
    novelty = F.lit(1.0) - F.col("best_sim")
    return scored.withColumn(out_col, F.col(deviation_col) * (F.lit(1.0) + F.lit(weight) * novelty))


def summarize(scored: DataFrame, ks: Sequence[int] = DEFAULT_KS) -> Dict[str, float]:
    """Quantiles of each aggregation, to see how differently they rank."""
    cols: List[str] = ["score_sum", "z_max", *[f"score_top{k}" for k in ks]]
    out: Dict[str, float] = {}
    for c in cols:
        q = scored.approxQuantile(c, [0.5, 0.9, 0.99], 0.005)
        out[f"{c}_p50"], out[f"{c}_p90"], out[f"{c}_p99"] = q
    return out


__all__ = [
    "column_moments",
    "deviation_scores",
    "score_objects",
    "unit_vector",
    "route",
    "novelty_column",
    "combine",
    "summarize",
    "DEFAULT_KS",
]


def column_quantiles(
    model: DataFrame, values: DataFrame, columns: Sequence[str]
) -> DataFrame:
    """Per cluster and column: the sorted member values, i.e. an empirical CDF.

    Returns (cluster_id, col, vals, n) with `vals` ascending.

    Kept as the values themselves rather than summarised into a mean and a
    standard deviation, because summarising is exactly what goes wrong on these
    columns. `kwh` and `irradiamento` are strongly skewed with a mass at zero,
    and a z-score on a skewed distribution is a poor measure of how unusual a
    value is. The concrete failure was visible in the injected faults: a `drop`
    pulls the target *toward* the mean, so its z goes down and a z-based score
    ranks it as more normal than the value it replaced -- the global reference
    put drops at ROC-AUC 0.41, below chance. A rank says the same value sits in
    the fifth percentile of its cluster, which is informative and assumes
    nothing about the shape.

    Cost is the cluster size per entry, a hundred or so doubles, which is why
    this is affordable at all: the array is joined to each scored row, so it
    wants to stay small.
    """
    clustered = model.filter(F.col("cluster_id") != F.lit(NOISE_CLUSTER_ID))
    joined = clustered.select("id", "cluster_id").join(values, on="id", how="inner")
    long = joined.select(
        "cluster_id",
        F.explode(
            F.array(*[F.struct(F.lit(c).alias("col"), F.col(c).cast("double").alias("v"))
                      for c in columns])
        ).alias("cv"),
    ).select("cluster_id", F.col("cv.col").alias("col"), F.col("cv.v").alias("v"))

    return long.groupBy("cluster_id", "col").agg(
        F.array_sort(F.collect_list("v")).alias("vals"),
        F.count("*").alias("n"),
    )


def quantile_scores(
    routed: DataFrame,
    values: DataFrame,
    quantiles: DataFrame,
    columns: Sequence[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> DataFrame:
    """How deep in its cluster's own distribution each column value sits.

        q     = fraction of cluster members at or below the value
        depth = 2 * |q - 0.5|      in [0, 1]

    Zero at the cluster's median for that column, one at either extreme. Both
    tails count: a value far below its cluster is as much an anomaly as one far
    above, and this is what a z-score misses on a skewed column.

    `depth` is bounded, which has a practical consequence worth stating: no
    column can dominate the aggregation through a badly estimated scale. A
    z-score can, and does -- a cluster of a hundred members whose sigma for one
    column came out spuriously small yields an enormous z, and the maximum and
    top-k aggregations are precisely the ones a single such column runs away
    with. Being bounded, depth makes the aggregations comparable to each other
    in a way the z versions are not.

    Returns (id, cluster_id, best_sim, q_max, q_top{k}..., q_sum, n_columns).
    """
    long = routed.select("id", "cluster_id", "best_sim").join(
        values.select("id", *columns), on="id", how="inner"
    ).select(
        "id", "cluster_id", "best_sim",
        F.explode(
            F.array(*[F.struct(F.lit(c).alias("col"), F.col(c).cast("double").alias("v"))
                      for c in columns])
        ).alias("cv"),
    ).select("id", "cluster_id", "best_sim",
             F.col("cv.col").alias("col"), F.col("cv.v").alias("v"))

    with_q = (
        long.join(quantiles, on=["cluster_id", "col"], how="left")
        .withColumn(
            # The empirical CDF, counted directly against the member values.
            # A test value below every member gives q = 0 and depth = 1, which
            # is the intended reading: nothing in this cluster is that low.
            "n_le",
            F.size(F.filter(F.coalesce(F.col("vals"), F.array()),
                            lambda y: y <= F.col("v"))),
        )
        .withColumn(
            "q",
            F.when(F.col("n") > 0, F.col("n_le") / F.col("n")).otherwise(F.lit(0.5)),
        )
        .withColumn("depth", F.lit(2.0) * F.abs(F.col("q") - F.lit(0.5)))
    )

    gathered = with_q.groupBy("id", "cluster_id", "best_sim").agg(
        F.reverse(F.array_sort(F.collect_list(F.col("depth") * F.col("depth")))).alias("d2"),
        F.max("depth").alias("q_max"),
        F.count("*").alias("n_columns"),
    )

    def rss(arr):
        return F.sqrt(F.aggregate(arr, F.lit(0.0), lambda acc, x: acc + x))

    out = gathered.withColumn("q_sum", rss(F.col("d2")))
    for k in ks:
        out = out.withColumn(f"q_top{k}", rss(F.slice(F.col("d2"), 1, k)))
    return out.drop("d2")
