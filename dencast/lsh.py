"""Step 1 of DENCAST: building the approximate neighborhood graph with LSH.

This follows the scheme described in the paper (section "Identification of the
neighborhood graph"), which in turn is the method of Ravichandran et al.:

    1. generate r random (m+k)-dimensional hyperplanes;
    2. represent each object as an r-bit signature, one bit per hyperplane,
       telling which side of that hyperplane the object falls on;
    3. generate numPerm random permutations of the r positions; for each one,
       permute every signature and sort the signatures lexicographically;
    4. for each object take its B positional neighbours in every sorted list,
       and keep a candidate pair only if its exact cosine similarity is at
       least minSim.

Why the permutations are needed: lexicographic sorting is dominated by the
leading bits, so two objects that differ only in the first bit end up at
opposite ends of the list even though their Hamming distance is 1. Permuting
the bit positions changes which bits lead, and with enough permutations every
similar pair eventually gets an ordering that pushes its differing bits to the
tail, making the two objects adjacent.

Complexity is O(|V| * m) instead of the O(|V|^2 * m) of an exact
neighbourhood computation.

Everything here is expressed in Spark SQL rather than with RDD lambdas or
Python UDFs, so the whole step runs inside the JVM with no serialization to a
Python interpreter.
"""

import logging
from typing import List, Tuple

import numpy as np
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from dencast.spark_session import cosine_similarity, dot
from dencast.utils import Params

logger = logging.getLogger(__name__)


def _random_hyperplanes(n_features: int, r: int, seed: int) -> np.ndarray:
    """Draw r random hyperplanes as Gaussian vectors, shape (n_features, r).

    Gaussian entries give directions uniformly distributed on the sphere,
    which is what makes the collision probability a function of the angle
    between two objects: P(same bit) = 1 - theta / pi.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_features, r))


def _random_permutations(r: int, num_permutations: int, seed: int) -> List[List[int]]:
    """Draw num_permutations random permutations of the r bit positions."""
    rng = np.random.default_rng(seed + 1)
    return [rng.permutation(r).tolist() for _ in range(num_permutations)]


def compute_signatures(
    df: DataFrame, vec_col: str, n_features: int, r: int, seed: int
) -> DataFrame:
    """Add a `bits` column holding the r-bit LSH signature of `vec_col`.

    The signature is stored as array<int> of 0/1, so that a permutation later
    becomes a plain reordering of element accesses.

    Each hyperplane is inlined in the query plan as a literal array, which
    keeps the projection inside Catalyst. The plan therefore holds r * m
    literals; with very wide feature vectors and a large r this makes plan
    compilation slower, though it does not affect the runtime per row.
    """
    planes = _random_hyperplanes(n_features, r, seed)

    bit_exprs = []
    for j in range(r):
        plane = F.array(*[F.lit(float(v)) for v in planes[:, j]])
        # One bit per hyperplane: which side of it the object falls on.
        bit_exprs.append(
            F.when(dot(F.col(vec_col), plane) >= F.lit(0.0), F.lit(1)).otherwise(F.lit(0))
        )

    return df.withColumn("bits", F.array(*bit_exprs))


def _permuted_signature_col(permutation: List[int]):
    """Build the permuted signature as a string.

    Reading the answers in a different order is all a permutation does: the
    object does not change, only the order in which its bits are written down.
    A string is used because Spark's own ordering on strings is exactly the
    lexicographic ordering the method calls for.
    """
    return F.concat_ws("", *[F.col("bits")[pos].cast("string") for pos in permutation])


def _candidate_pairs_for_permutation(
    df_bits: DataFrame, permutation: List[int], b: int
) -> DataFrame:
    """Return the candidate pairs produced by one permutation, as (a, b).

    After sorting the permuted signatures, each object is paired with the b
    objects that follow it in the list. Since the pairing is symmetric, doing
    it in one direction only already gives every object its b positional
    neighbours on both sides.

    Implementation note: `row_number` over an unpartitioned window moves all
    rows to a single partition, which is the one point in this port that does
    not scale the way the Scala version does. On a real cluster the sorted
    RDD would be numbered with `zipWithIndex` instead, which keeps the data
    distributed; that path needs Python workers, which is why the SQL form is
    used here.
    """
    indexed = (
        df_bits.withColumn("sig", _permuted_signature_col(permutation))
        # Ordered by signature *and then by id*. The signature alone is not a
        # total order: with r bits there are only 2^r of them, so at r=5 some
        # three hundred objects in a ten-thousand-object window share one, and
        # their relative position is decided by whatever order the shuffle
        # happened to produce. The sliding window then picks b of them at
        # random, differently on every run -- two fits of the same data with
        # the same seed came out with 564 and 665 clusters. The id breaks the
        # tie deterministically and costs nothing.
        .withColumn("idx", F.row_number().over(Window.orderBy("sig", "id")) - F.lit(1))
        .select("id", "idx")
    )

    # Expanding the offsets 1..b turns the "b nearest positions" condition into
    # an equi-join, which Spark can plan efficiently, instead of a range join.
    left = (
        indexed.select(F.col("id").alias("id_a"), F.col("idx").alias("idx_a"))
        .withColumn("offset", F.explode(F.sequence(F.lit(1), F.lit(b))))
        .withColumn("idx_b", F.col("idx_a") + F.col("offset"))
        .select("id_a", "idx_b")
    )
    right = indexed.select(F.col("id").alias("id_b"), F.col("idx").alias("idx_b"))

    pairs = left.join(right, on="idx_b", how="inner").select("id_a", "id_b")

    # Normalize the orientation so that (u, v) and (v, u) are the same pair.
    return pairs.select(
        F.least("id_a", "id_b").alias("a"),
        F.greatest("id_a", "id_b").alias("b"),
    ).filter(F.col("a") != F.col("b"))


def build_neighborhood_graph(
    spark: SparkSession, labeled: DataFrame, params: Params, seed: int
) -> Tuple[DataFrame, DataFrame]:
    """Build the approximate neighborhood graph.

    Args:
        labeled: must contain `id` (long), `features` (array<double>, the m
            descriptive attributes) and `targets` (array<double>, the k target
            attributes).
        params: the project parameters.
        seed: seed for the hyperplanes and the permutations.

    Returns:
        (vertices, edges) where vertices is (id) and edges is (src, dst, sim)
        with both directions present, as in the original implementation.
    """
    lsh_cfg = params.lsh

    # The paper hashes the full (m + k)-dimensional vector, so the graph
    # reflects similarity in inputs *and* outputs. Turning this off builds the
    # graph on the descriptive attributes only, which is what an inductive
    # mapping of new unlabeled objects would require.
    if lsh_cfg.use_targets_in_signature:
        df = labeled.withColumn("lsh_vec", F.concat(F.col("features"), F.col("targets")))
        n_features = params.dataset.m + params.dataset.k
    else:
        df = labeled.withColumn("lsh_vec", F.col("features"))
        n_features = params.dataset.m

    df = df.select("id", "lsh_vec").repartition(params.spark.num_partitions).cache()
    n_objects = df.count()
    logger.info(
        "Building the neighborhood graph over %d objects (%d-dimensional)",
        n_objects,
        n_features,
    )

    # The hyperplanes have n_features rows, so a vector of a different length
    # produces a dot product over the shorter of the two -- silently, because
    # Spark's zip_with simply stops. The result is a plausible-looking signature
    # computed from part of the object, which is far worse than a crash: it was
    # exactly this that let a stale 16-column parquet run against a 17-column
    # configuration and report numbers.
    actual = df.select(F.size("lsh_vec").alias("n")).first()["n"]
    if actual != n_features:
        raise ValueError(
            f"lsh_vec has {actual} components but the configuration implies "
            f"{n_features} (dataset.m={params.dataset.m}, k={params.dataset.k}). "
            "The processed parquet is stale: rerun the features stage."
        )

    df_bits = compute_signatures(df, "lsh_vec", n_features, lsh_cfg.r, seed).select(
        "id", "bits"
    )
    df_bits = df_bits.cache()
    df_bits.count()

    permutations = _random_permutations(lsh_cfg.r, lsh_cfg.num_permutations, seed)

    # Materialised every `chunk` permutations rather than unioned into a single
    # plan. Each permutation carries a `row_number` over an unpartitioned window,
    # which Spark can only evaluate by gathering the whole set into one task, plus
    # the r * m hyperplane literals inlined in the projection. Thirty of those in
    # one query means thirty live sort buffers with nothing allowed to release
    # them, and on a 3 GB heap the fit then fails -- nondeterministically, which
    # is worse than failing outright: the same 38,000-object fit completed in 118
    # seconds and then exhausted the heap on the next identical run.
    # Checkpointing cuts the lineage so each chunk's memory is reclaimed before
    # the next begins. It changes no result.
    chunk = 5
    candidates = None
    pending = None
    for i, permutation in enumerate(permutations, start=1):
        part = _candidate_pairs_for_permutation(df_bits, permutation, lsh_cfg.b)
        pending = part if pending is None else pending.unionByName(part)
        if i % chunk == 0 or i == len(permutations):
            # Deduplicated at each checkpoint, not only at the end: the same pair
            # recurs across permutations, so carrying the copies forward makes the
            # accumulated set grow faster than it needs to.
            materialised = pending.dropDuplicates(["a", "b"]).localCheckpoint(eager=True)
            candidates = (
                materialised
                if candidates is None
                else candidates.unionByName(materialised)
                .dropDuplicates(["a", "b"])
                .localCheckpoint(eager=True)
            )
            pending = None
            logger.debug("Permutations up to %d/%d materialised", i, len(permutations))

    candidates = candidates.cache()
    n_candidates = candidates.count()
    logger.info(
        "Candidate pairs after %d permutations: %d (%.1f per object)",
        len(permutations),
        n_candidates,
        n_candidates / n_objects if n_objects else 0.0,
    )

    # Exact verification. LSH only proposes candidates; the true cosine is what
    # decides whether an edge exists. This is the filter that removes false
    # positives, so the resulting graph has missing edges but no spurious ones.
    vecs_a = df.select(F.col("id").alias("a"), F.col("lsh_vec").alias("vec_a"))
    vecs_b = df.select(F.col("id").alias("b"), F.col("lsh_vec").alias("vec_b"))

    verified = (
        candidates.join(vecs_a, on="a")
        .join(vecs_b, on="b")
        .withColumn("sim", cosine_similarity(F.col("vec_a"), F.col("vec_b")))
        .filter(F.col("sim") >= F.lit(lsh_cfg.min_sim))
        .select("a", "b", "sim")
        .cache()
    )
    n_edges_und = verified.count()
    logger.info(
        "Undirected edges surviving minSim=%.3f: %d (%.1f%% of candidates)",
        lsh_cfg.min_sim,
        n_edges_und,
        100.0 * n_edges_und / n_candidates if n_candidates else 0.0,
    )

    # Both directions, mirroring `flatMap(x => List((x.i, x.j), (x.j, x.i)))`
    # in the Scala implementation. The neighborhood relation is symmetric, and
    # having both directions makes the degree count and the message passing
    # straightforward.
    edges = verified.select(
        F.col("a").alias("src"), F.col("b").alias("dst"), F.col("sim")
    ).unionByName(
        verified.select(F.col("b").alias("src"), F.col("a").alias("dst"), F.col("sim"))
    )

    # Truncate the lineage: from here on the graph is a plain table, and
    # nothing downstream has to re-derive the permutation branches.
    edges = edges.localCheckpoint(eager=True)
    vertices = df.select("id").localCheckpoint(eager=True)

    candidates.unpersist()
    verified.unpersist()

    return vertices, edges


def cross_candidate_pairs_from_bits(
    q_bits: DataFrame,
    r_bits: DataFrame,
    params: Params,
    seed: int,
    chunk: int = 5,
) -> DataFrame:
    """Candidate (query, reference) pairs from signatures already computed.

    `build_neighborhood_graph` blocks one set against itself. Routing needs one
    set against another: for every object to be scored, which reference objects
    are worth an exact cosine. The alternative is the full product, which on
    this dataset is 43,000 x 162,000 = seven billion comparisons.

    **Both sides must have been hashed with the same hyperplanes**, i.e. the
    same `r` and the same seed. Signatures from different planes would not be
    comparable, and the bug would be silent: the sort would still run and still
    emit pairs, just meaningless ones. Taking bits rather than vectors is what
    makes that explicit, and it lets the caller hash a large reference set once
    and reuse it across many query batches instead of recomputing the expensive
    side every time.

    The window runs in **both** directions, unlike the graph's. There each
    object takes the b positions after it and pairs are made undirected, so a
    neighbour behind you finds you when its own turn comes. Here the pairs are
    directed -- only query-to-reference survives -- so looking forward alone
    would discard every reference object that happened to sort before the query.
    That doubles the candidates per object to `2 * b * num_permutations`, which
    is worth keeping in mind when sizing a batch: at b=30 and 30 permutations it
    is 1,800 per object before deduplication, not 900.

    Returns (q_id, r_id), deduplicated. An object with no candidate is simply
    absent, which the caller must handle: that is this method's recall loss and
    it should be counted rather than hidden.
    """
    lsh_cfg = params.lsh
    pool = q_bits.select("id", "bits", F.lit(True).alias("is_q")).unionByName(
        r_bits.select("id", "bits", F.lit(False).alias("is_q"))
    )

    # The same seed the reference's signatures were computed with: the
    # permutations only reorder bits, but query and reference must be reordered
    # identically or the sort compares differently-written signatures.
    #
    # Materialised every `chunk` permutations rather than unioned into one plan.
    # Each permutation carries a `row_number` over an unpartitioned window --
    # which Spark can only evaluate by collecting the whole pool into a single
    # task -- plus the r * m hyperplane literals inlined in the projection.
    # Thirty of those in one query gives a multi-megabyte task binary and thirty
    # live sort buffers with nothing allowed to release them, which is what
    # exhausted the heap. Checkpointing cuts the lineage so each chunk's memory
    # can be reclaimed before the next begins.
    permutations = _random_permutations(lsh_cfg.r, lsh_cfg.num_permutations, seed)
    accumulated = None
    pending = None

    for i, permutation in enumerate(permutations, start=1):
        indexed = pool.withColumn("sig", _permuted_signature_col(permutation)).withColumn(
            # Sorted by signature and then by id, for the same reason the graph
            # does: with r bits there are only 2^r signatures, so ties are the
            # normal case, and leaving their order to the shuffle would make the
            # whole result irreproducible.
            "idx",
            F.row_number().over(Window.orderBy("sig", "id")) - F.lit(1),
        )
        left = (
            indexed.filter(F.col("is_q"))
            .select(F.col("id").alias("q_id"), F.col("idx").alias("idx_q"))
            .withColumn(
                "offset",
                F.explode(
                    F.array_except(
                        F.sequence(-F.lit(lsh_cfg.b), F.lit(lsh_cfg.b)), F.array(F.lit(0))
                    )
                ),
            )
            .withColumn("idx_r", F.col("idx_q") + F.col("offset"))
            .select("q_id", "idx_r")
        )
        right = indexed.filter(~F.col("is_q")).select(
            F.col("id").alias("r_id"), F.col("idx").alias("idx_r")
        )
        part = left.join(right, on="idx_r", how="inner").select("q_id", "r_id")
        pending = part if pending is None else pending.unionByName(part)

        if i % chunk == 0 or i == len(permutations):
            # Deduplicated at every checkpoint, not only at the end: the same
            # pair recurs across permutations, and carrying the copies forward
            # is what makes the accumulated set grow faster than it needs to.
            materialised = pending.dropDuplicates(["q_id", "r_id"]).localCheckpoint(
                eager=True
            )
            accumulated = (
                materialised
                if accumulated is None
                else accumulated.unionByName(materialised)
                .dropDuplicates(["q_id", "r_id"])
                .localCheckpoint(eager=True)
            )
            pending = None

    if accumulated is None:
        raise ValueError("Nessuna permutazione: lsh.num_permutations e zero?")
    return accumulated
