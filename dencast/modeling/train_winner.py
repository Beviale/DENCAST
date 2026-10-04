"""Stage: fit the selected configuration on train+valid and read test once.

`train_and_select` chose a configuration and a formula by scoring the grid on
valid. This refits that choice on train and valid together -- more data, since
nothing is being selected any more -- and reports the number on test. It reads
`selection.json` rather than re-deriving the winner, so the configuration
reported is provably the one that was chosen and not a fresh argmax taken over
test.

**Every formula is still reported, and only one of them is the result.** The
others are context: they say whether the winner won by a margin or by a rounding
error. Quoting the best test number instead would be selection on test wearing a
different hat, and would inflate the figure by however many candidates the grid
held.

Usage:
    uv run python -m dencast.modeling.train_winner
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
import typer

from dencast.anomaly_metrics import evaluate_ranking, format_ranking
from dencast.config import PROCESSED_DATA_DIR, REPORTS_DIR
from dencast.modeling.detect import (
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
    selection_path: Path = typer.Option(Path("reports/selection.json")),
    max_test_days: Optional[int] = typer.Option(None, help="Cap the test days scored"),
    kinds_csv: Optional[Path] = typer.Option(
        None, help="Source CSV, for the per-kind breakdown"
    ),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Score the test split with the configuration valid chose."""
    params = Params.load(params_path)
    params.validate()
    seed = params.evaluation.seeds[0]
    out = output or REPORTS_DIR / "test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    if not selection_path.exists():
        raise typer.BadParameter(
            f"{selection_path} non esiste: esegui prima train_and_select"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if "winner" not in selection:
        raise typer.BadParameter(
            f"{selection_path} non ha un vincitore: la selezione non e arrivata in fondo"
        )
    winner = selection["winner"]
    logger.info("Vincitore da valid: {} con formula '{}' ({} {:.4f})",
                winner["overrides"] or "default", winner["formula"],
                winner["metric"], winner["value"])

    final = apply_overrides(params, winner["overrides"])
    logger.info("Colonne: {} nel routing, {} nella deviazione",
                len(final.dataset.feature_cols), len(deviation_columns(final)))

    spark = get_spark(final)
    try:
        df = spark.read.parquet(
            str(PROCESSED_DATA_DIR / f"{final.dataset.name}.parquet")
        ).cache()
        values = build_values(spark, final).cache()
        dates = df.select("id", "date").cache()

        days = split_days_3(df, final)
        # Selection is over, so valid joins train: there is nothing left to
        # choose and holding data back would only make the final model worse
        # than the one the numbers are meant to describe.
        fit_days = list(days.train) + list(days.valid)
        # Same memory bound as the selection, applied to the same end of the
        # history so the two stages fit on comparable amounts of data. Capping
        # only one of them would confound the configuration's effect with how
        # much data it was given.
        if max_train_days:
            fit_days = fit_days[-(max_train_days + len(days.valid)):]
        test = days.test[:max_test_days] if max_test_days else days.test
        logger.info("Fit su {} giorni (train+valid), test su {} giorni",
                    len(fit_days), len(test))

        fitted = fit_split(spark, df, final, fit_days, seed)
        try:
            scores, labels, ids, diag = score_days(
                spark, fitted.model, values, final, test, dates, seed
            )
        finally:
            fitted.model.unpersist()
            fitted.edges.unpersist()

        per_formula = evaluate_all(scores, labels)
        result: Dict = {
            "winner": winner,
            "n_fit_days": len(fit_days),
            "n_test_days": len(test),
            "clustering": {
                k: fitted.stats[k] for k in ("n_clusters", "n_core", "n_noise", "n_edges")
            },
            "diagnostics": diag,
            "headline": {"formula": winner["formula"], **per_formula[winner["formula"]]},
            "all_formulas": per_formula,
        }

        # Per kind, when the source is at hand. A single pooled number hides the
        # most informative thing about this formulation: it is strong on exactly
        # the faults a residual score misses and weak on the ones it catches, so
        # the pooled figure averages two opposite regimes.
        if kinds_csv and kinds_csv.exists():
            import pandas as pd

            src = pd.read_csv(kinds_csv, usecols=["id", "anomaly_kind"])
            kinds = dict(zip(src["id"], src["anomaly_kind"].fillna("")))
            best = scores[winner["formula"]]
            by_kind: Dict[str, Dict] = {}
            for kind in sorted(set(kinds.values()) - {""}):
                # Negatives are kept whole and only the other kinds' positives
                # dropped, so each number is "this kind against normal traffic"
                # at the real base rate rather than against a rebalanced set.
                keep = [
                    j for j, (i_, lab) in enumerate(zip(ids, labels))
                    if lab == 0 or kinds.get(i_, "") == kind
                ]
                if any(labels[j] for j in keep):
                    by_kind[kind] = evaluate_ranking(
                        [best[j] for j in keep], [labels[j] for j in keep], ks=(50,)
                    )
            result["by_kind"] = by_kind

        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

        logger.success("")
        logger.success("  === test: {} oggetti, {} anomali ({:.2%}) ===",
                       f"{int(diag['n_scored']):,}", int(diag["n_anomalies"]),
                       diag["n_anomalies"] / diag["n_scored"])
        logger.success("")
        logger.success("    RISULTATO ({}): {}", winner["formula"],
                       format_ranking(per_formula[winner["formula"]]))
        logger.success("")
        logger.success("  === le altre formule, per contesto ===")
        for f, m in sorted(per_formula.items(), key=lambda kv: -kv[1]["average_precision"]):
            mark = "  <-- scelta su valid" if f == winner["formula"] else ""
            logger.success("    {:<16} {}{}", f, format_ranking(m), mark)
        logger.success("")
        logger.success("    tasso base: {:.4f}", diag["n_anomalies"] / diag["n_scored"])
        logger.success("    cluster: {}  rumore: {}",
                       result["clustering"]["n_clusters"], result["clustering"]["n_noise"])
        logger.success("    non instradati da LSH: {:.2%}", diag["frac_unrouted"])
        logger.success("    saturazione: {:.2%} degli score a 1.0 esatto",
                       diag["frac_sigmoid_at_one"])
        if result.get("by_kind"):
            logger.success("")
            logger.success("  === per tipo di anomalia ===")
            for kind, m in sorted(
                result["by_kind"].items(), key=lambda kv: -kv[1]["average_precision"]
            ):
                logger.success(
                    "    {:<7} AUC-PR {:.4f}   ROC-AUC {:.4f}   su {} casi",
                    kind, m["average_precision"], m["roc_auc"], int(m["n_anomalies"]),
                )
    finally:
        spark.stop()


if __name__ == "__main__":
    app()
