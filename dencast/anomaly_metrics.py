"""Metrics for a ranking of objects by anomaly score.

The model produces an ordering, not a decision: nothing here picks a
threshold. These metrics ask how well the ordering separates the labelled
anomalies from the rest, which is the question a ranking can answer.

**Why AUC-PR rather than accuracy or ROC AUC alone.** Anomalies are rare, so a
detector that flags nothing scores 99%+ accuracy and is useless. ROC AUC
survives that, but with a base rate of 1% it still looks flattering: it
averages over thresholds nobody would ever use, where the detector flags half
the dataset. Average precision follows the precision actually obtained at each
recall level, so it degrades when the top of the ranking fills with normal
objects -- exactly the failure that matters.

**Why the base rate is reported alongside.** An average precision of 0.30 is
poor at a 20% base rate and excellent at 0.5%. The number means nothing on its
own; `lift` in the summary is the honest version, precision@k divided by what
guessing would give.

**Ties are handled explicitly.** Scores tie often here -- objects in the same
small cluster share a sigma, and g is a step function, so identical scores are
common rather than freak events. Both AUC computations treat a group of equal
scores as one threshold, so the result does not depend on the order the rows
happened to arrive in. A metric that quietly rewards a favourable tie order is
a metric that improves when you change nothing.

Everything here runs on the driver over plain sequences: a ranking metric
needs a total order, so there is no distributed formulation that avoids
bringing the scores together. The inputs are one row per evaluated object --
report scale, not dataset scale.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from loguru import logger


def _sorted_pairs(
    scores: Sequence[float], labels: Sequence[int]
) -> List[Tuple[float, int]]:
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels differ in length: {len(scores)} vs {len(labels)}"
        )
    return sorted(zip(scores, labels), key=lambda p: -p[0])


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the precision-recall curve (AUC-PR).

    Integrated over distinct score thresholds rather than over rows: every
    object with the same score is above or below a given threshold together,
    so a tied group contributes a single point on the curve. Walking row by
    row instead would let the arbitrary order inside a tie change the answer.

    Returns 0.0 when there is no positive to find.
    """
    pairs = _sorted_pairs(scores, labels)
    n_pos = sum(1 for _, y in pairs if y)
    if n_pos == 0:
        return 0.0

    ap = 0.0
    tp = seen = 0
    prev_recall = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            seen += 1
            j += 1
        recall = tp / n_pos
        precision = tp / seen
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return ap


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve, via the rank-sum identity.

    Equivalent to the probability that a randomly chosen anomaly outranks a
    randomly chosen normal object. Tied scores share the average of the ranks
    they span, which is what makes a tie count as half a win rather than a
    whole one.

    Returns 0.5 -- the value of guessing -- when either class is absent.
    """
    pairs = _sorted_pairs(scores, labels)
    n = len(pairs)
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Ascending ranks 1..n, averaged within each group of equal scores.
    ranks = [0.0] * n
    ascending = list(reversed(pairs))
    i = 0
    while i < n:
        j = i
        while j < n and ascending[j][0] == ascending[i][0]:
            j += 1
        shared = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = shared
        i = j

    rank_sum = sum(r for r, (_, y) in zip(ranks, ascending) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def precision_at_k(scores: Sequence[float], labels: Sequence[int], k: int) -> float:
    """Share of the top k that are genuine anomalies.

    The operational metric: if an analyst can review k objects a day, this is
    how much of that budget is well spent. A tie straddling position k is cut
    arbitrarily, which is unavoidable -- the ranking genuinely does not order
    those objects.
    """
    if k <= 0:
        return 0.0
    pairs = _sorted_pairs(scores, labels)[:k]
    return sum(y for _, y in pairs) / len(pairs) if pairs else 0.0


def evaluate_ranking(
    scores: Sequence[float], labels: Sequence[int], ks: Sequence[int] = (10, 50, 100)
) -> Dict[str, float]:
    """Every metric above, plus the context needed to read them.

    `lift_at_k` is precision@k divided by the base rate: how many times better
    than guessing. It is the number to quote when the base rate is unusual,
    because precision alone is not comparable across datasets.
    """
    n = len(scores)
    n_pos = sum(1 for y in labels if y)
    base_rate = n_pos / n if n else 0.0

    out: Dict[str, float] = {
        "n_scored": float(n),
        "n_anomalies": float(n_pos),
        "base_rate": base_rate,
        "average_precision": average_precision(scores, labels),
        "roc_auc": roc_auc(scores, labels),
    }
    for k in ks:
        if k > n:
            continue
        p = precision_at_k(scores, labels, k)
        out[f"precision_at_{k}"] = p
        out[f"lift_at_{k}"] = p / base_rate if base_rate else 0.0

    # Precision at the number of anomalies actually present: the ceiling is 1.0
    # and reaching it means a perfect ranking, which makes it readable without
    # knowing the base rate.
    if n_pos:
        out["precision_at_n"] = precision_at_k(scores, labels, n_pos)
    return out


def format_ranking(metrics: Dict[str, float]) -> str:
    """One readable line, for logs."""
    parts = [
        f"AUC-PR {metrics.get('average_precision', 0):.4f}",
        f"ROC-AUC {metrics.get('roc_auc', 0):.4f}",
        f"base rate {metrics.get('base_rate', 0):.2%}",
        f"{int(metrics.get('n_anomalies', 0))}/{int(metrics.get('n_scored', 0))} anomalies",
    ]
    if "precision_at_n" in metrics:
        parts.insert(2, f"P@n {metrics['precision_at_n']:.4f}")
    return "  ".join(parts)


def evaluate_scored(df, score_col: str = "score_max", label_col: str = "anomaly"):
    """Collect a scored DataFrame and evaluate the ranking it defines.

    Raises when the label column is absent rather than returning an empty
    result: reaching here means the caller believed the data was labelled, and
    a silent zero would be indistinguishable from a detector that found
    nothing.
    """
    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' is not in the scored data (have: "
            f"{', '.join(df.columns)}). Set dataset.anomaly_col to null if "
            "this dataset has no labels."
        )
    rows = df.select(score_col, label_col).collect()
    scores = [float(r[score_col]) for r in rows]
    labels = [1 if r[label_col] else 0 for r in rows]
    metrics = evaluate_ranking(scores, labels)
    logger.info("Ranking: {}", format_ranking(metrics))
    return metrics
