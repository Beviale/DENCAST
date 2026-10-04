"""Stage: choose a configuration by fitting on train and scoring valid.

One fit per grid point on the whole train split, then every object in valid is
scored by how far its columns sit from its cluster. The configuration that ranks
anomalies best on valid wins. Test is not opened here.

No rolling window. The split is chronological, so all of train precedes all of
valid and there is nothing left for a window to protect against.

**The formula is selected too, not fixed.** Four rankings come out of every fit
-- the two sigmoid variants and their untransformed counterparts -- and which to
use is as much a choice as `lsh.r` is. Picking it by looking at test would be the
same leak as tuning a hyperparameter there, so it is chosen on valid alongside
everything else and reported as part of the winner.

**Nothing here reads a label except the metric.** Every model fitted is
unsupervised: it sees columns and clusters, never the `anomaly` column. Valid's
labels enter only to rank the candidates, which makes this supervised model
*selection* over unsupervised models -- worth naming, because it is the one place
a label touches the search.

Usage:
    uv run python -m dencast.modeling.train_and_select
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
import typer

from dencast.anomaly_metrics import format_ranking
from dencast.config import PROCESSED_DATA_DIR, REPORTS_DIR
from dencast.modeling.detect import (
    FORMULAS,
    apply_overrides,
    build_values,
    deviation_columns,
    evaluate_all,
    fit_split,
    score_days,
)
from dencast.spark_session import get_spark
from dencast.split import split_days_3
from dencast.utils import Params

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    params_path: Path = typer.Option(Path("params.yaml")),
    max_train_days: Optional[int] = typer.Option(
        None,
        help="Fit on only the most recent N days of the fitting split. A memory "
             "bound, not a modelling choice: see the note where it is applied.",
    ),
    max_valid_days: Optional[int] = typer.Option(
        None, help="Cap the valid days scored, for a quick look"
    ),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Score every grid point on valid and record the winner."""
    params = Params.load(params_path)
    params.validate()
    seed = params.evaluation.seeds[0]
    metric = params.selection.metric
    out = output or REPORTS_DIR / "selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    combos = params.selection.combinations()
    logger.info(
        "Griglia: {} configurazioni x {} formule = {} candidati",
        len(combos), len(FORMULAS), len(combos) * len(FORMULAS),
    )
    logger.info("Colonne: {} nel routing, {} nella deviazione",
                len(params.dataset.feature_cols), len(deviation_columns(params)))

    spark = get_spark(params)
    try:
        df = spark.read.parquet(
            str(PROCESSED_DATA_DIR / f"{params.dataset.name}.parquet")
        ).cache()
        values = build_values(spark, params).cache()
        dates = df.select("id", "date").cache()

        days = split_days_3(df, params)
        valid = days.valid[:max_valid_days] if max_valid_days else days.valid
        # The *most recent* N days, not a random sample of them. Both shrink the
        # fit; only one preserves what the clustering depends on. Sampling rows
        # uniformly across 512 days thins the data everywhere, and since min_pts
        # and min_sim are absolute thresholds on a neighbourhood, a set 4x
        # sparser makes almost everything noise -- the clustering would fail for
        # a reason that has nothing to do with the method being tested. Taking a
        # contiguous tail keeps the density within a day untouched and only
        # shortens the history, and the recent days are the ones adjacent to
        # valid anyway.
        train = days.train[-max_train_days:] if max_train_days else days.train
        logger.info("Fit su {} giorni di train{}, scoring su {} giorni di valid",
                    len(train),
                    f" (gli ultimi, di {len(days.train)})" if max_train_days else "",
                    len(valid))

        results: List[Dict] = []
        for i, overrides in enumerate(combos, start=1):
            logger.info("=== configurazione {}/{}: {} ===",
                        i, len(combos), overrides or "default")
            candidate = apply_overrides(params, overrides)
            fitted = fit_split(spark, df, candidate, train, seed)
            try:
                scores, labels, _ids, diag = score_days(
                    spark, fitted.model, values, candidate, valid, dates, seed
                )
            finally:
                fitted.model.unpersist()
                fitted.edges.unpersist()
            per_formula = evaluate_all(scores, labels)
            results.append({
                "overrides": overrides,
                "clustering": {
                    k: fitted.stats[k] for k in ("n_clusters", "n_core", "n_noise", "n_edges")
                },
                "diagnostics": diag,
                "formulas": per_formula,
            })
            for f, m in sorted(per_formula.items(), key=lambda kv: -kv[1][metric]):
                logger.info("    {:<16} {}", f, format_ranking(m))
            # Written every iteration: a grid that dies on point seven should
            # still leave the first six behind.
            out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

        # --- the winner: a (configuration, formula) pair ----------------------
        best = max(
            ((r, f) for r in results for f in FORMULAS),
            key=lambda pair: pair[0]["formulas"][pair[1]][metric],
        )
        winner = {
            "overrides": best[0]["overrides"],
            "formula": best[1],
            "metric": metric,
            "value": best[0]["formulas"][best[1]][metric],
            "n_valid_days": len(valid),
            "n_candidates": len(combos) * len(FORMULAS),
        }
        out.write_text(
            json.dumps({"winner": winner, "results": results}, indent=2), encoding="utf-8"
        )

        logger.success("")
        logger.success("  === vincitore su valid ===")
        logger.success("    configurazione: {}", winner["overrides"] or "default")
        logger.success("    formula:        {}", winner["formula"])
        logger.success("    {}: {:.4f}", metric, winner["value"])
        d = best[0]["diagnostics"]
        logger.success("")
        logger.success("    cluster: {}  rumore: {}",
                       best[0]["clustering"]["n_clusters"], best[0]["clustering"]["n_noise"])
        logger.success("    non instradati da LSH: {:.2%}", d["frac_unrouted"])
        logger.success("    valori distinti: sigmoide {:.0f} vs grezzo {:.0f}",
                       d["distinct_sigmoid_plain"], d["distinct_sum_sq"])
        if d["distinct_sigmoid_plain"] < d["distinct_sum_sq"]:
            logger.warning(
                "    la sigmoide ha perso {:.0f} valori distinti: satura, e il suo "
                "ranking e degradato di altrettanto",
                d["distinct_sum_sq"] - d["distinct_sigmoid_plain"],
            )
    finally:
        spark.stop()


if __name__ == "__main__":
    app()
