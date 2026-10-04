"""Stage: rank objects by how anomalous their targets are.

For every evaluation day the model is fit on the window before it, the day is
predicted, and each object is scored by how far its error exceeds what is
normal for an object with its assignment confidence:

                    | y - prediction |
    score  =  --------------------------------
                sigma_loo(cluster)  *  g(d)

The sigma is computed with the object left out of its own cluster, because
with the model trained on everything each object helped form the statistics it
is judged against and would otherwise mask itself. g comes from the
`calibrate` stage and was fitted on days disjoint from these.

Both `score_max` and `z_max` are written out -- the score with and without the
g term. On data where the nearest-neighbour similarity saturates, g may add
little, and keeping both means that can be measured once labels exist rather
than assumed now.

No labels are needed to produce the ranking; they are needed to judge it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger
import mlflow
from pyspark.sql import functions as F
import typer

from dencast import tracking
from dencast.anomaly import DEFAULT_SHRINKAGE, full_scores
from dencast.anomaly_metrics import evaluate_ranking, evaluate_scored, format_ranking
from dencast.calibration import load_expected_residual
from dencast.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from dencast.data.features import hide_targets
from dencast.io import write_csv
from dencast.modeling.predict import attach_ground_truth, predict
from dencast.spark_session import get_spark
from dencast.utils import Params

app = typer.Typer(help=__doc__, add_completion=False)

BATCH_DAYS = 15
"""Test days scored per Spark job; see the loop in `main` for the trade-off."""


@app.command()
def main(
    input_path: Optional[Path] = typer.Option(None, help="Override the processed dataset"),
    curve_path: Optional[Path] = typer.Option(None, help="Override the calibration curve"),
    output: Optional[Path] = typer.Option(None, help="Override the scores output"),
    shrinkage: float = typer.Option(DEFAULT_SHRINKAGE, help="Variance prior pseudo-count"),
    top_n: int = typer.Option(20, help="How many top-scoring objects to log"),
) -> None:
    """DVC stage `score`: rank every evaluation object by anomaly score."""
    params = Params.load()
    params.validate()

    processed = input_path or PROCESSED_DATA_DIR / f"{params.dataset.name}_test.parquet"
    curve_file = curve_path or MODELS_DIR / "calibration" / f"{params.dataset.name}.json"
    model_file = MODELS_DIR / params.dataset.name / "model.parquet"
    out_path = output or REPORTS_DIR / "anomaly_scores.csv"
    summary_path = REPORTS_DIR / "anomaly_summary.json"

    curve = load_expected_residual(curve_file)
    logger.info("Loaded calibration curve: {}", curve)

    tracking.setup(params)
    spark = get_spark(params)

    try:
        df = spark.read.parquet(str(processed)).cache()
        dates = sorted(r["date"] for r in df.select("date").distinct().collect())

        run_name = f"{params.mlflow.run_name_prefix}-score"
        with tracking.start_run(
            params,
            run_name,
            tags={
                "stage": "score",
                "dataset": params.dataset.name,
                "n_days": len(dates),
                "shrinkage": shrinkage,
            },
        ):
            # One model for every day: the one `train_winner` fitted on the
            # whole train split. Refitting per day, as this stage used to,
            # would rank the anomalies with 358 different models while the
            # regression figures came from a single one -- two experiments
            # reported as one. It is also about ten times slower, because a
            # fit costs far more than routing a day through an existing model.
            #
            # leave_one_out stays False for the same reason as before: the
            # split is chronological, so no test object was ever inside the
            # training data whose cluster statistics judge it.
            model = spark.read.parquet(str(model_file)).cache()
            n_model = model.count()
            logger.info("Scoring against the fixed model: {} objects", n_model)

            # Batched, not one day at a time. A day holds about a hundred test
            # rows, so the arithmetic is trivial either way -- but each day
            # costs dozens of small Spark jobs whose scheduling dominates
            # everything else. Measured: 18 seconds a day one at a time, for
            # work that takes milliseconds. Fifteen days per batch is still
            # only ~52 million similarity pairs in a job, comfortably under
            # the 174 million that exhausted the heap, and cuts 358 rounds of
            # overhead to 24.
            dates_by_batch = [
                dates[i : i + BATCH_DAYS] for i in range(0, len(dates), BATCH_DAYS)
            ]
            day_of = df.select("id", "date")

            collected = None
            done = 0
            for batch in dates_by_batch:
                test = df.filter(F.col("date").isin(batch))
                if test.count() == 0:
                    continue

                scored = attach_ground_truth(
                    predict(hide_targets(test), model, params.dataset.k), test
                )
                # full_scores keys by id and drops the date, so it comes back
                # from the split rather than being pinned to a single day as
                # it could be when the loop handled one day at a time.
                batch_scores = full_scores(
                    model, scored, curve, shrinkage=shrinkage, leave_one_out=False
                ).join(day_of, on="id", how="left")

                collected = (
                    batch_scores
                    if collected is None
                    else collected.unionByName(batch_scores)
                )
                # Materialised every batch: nothing is computed until then, so
                # this is what bounds how much work lands in one job.
                collected = collected.localCheckpoint(eager=True)
                done += len(batch)
                logger.info("  {}/{} days  ({})", done, len(dates), batch[-1])

            if collected is None:
                raise typer.Exit(code=1)

            # The label joins here and nowhere earlier. Everything above --
            # the model, the calibration curve, the cluster statistics -- was
            # built without it, which is what makes the ranking an honest
            # prediction rather than a fit to the answer.
            label_col = params.dataset.anomaly_col
            if label_col:
                if "anomaly" not in df.columns:
                    raise typer.BadParameter(
                        f"dataset.anomaly_col is '{label_col}' but the test split has "
                        "no 'anomaly' column. Re-run split_dataset."
                    )
                collected = collected.join(
                    df.select("id", "anomaly"), on="id", how="left"
                ).withColumn("anomaly", F.coalesce(F.col("anomaly"), F.lit(0)))

            collected = collected.cache()
            n = collected.count()
            logger.info("Scored {} objects over {} days", n, len(dates))

            ranking: dict = {}
            if label_col:
                # Both rankings are evaluated: score_max divides by g, z_max
                # does not. Reporting them side by side is what turns "does
                # the calibration term earn its place?" into a number instead
                # of an argument.
                ranking = evaluate_scored(collected, "score_max", "anomaly")
                logger.success("score_max  {}", format_ranking(ranking))
                without_g = evaluate_ranking(
                    [float(r["z_max"]) for r in collected.select("z_max").collect()],
                    [1 if r["anomaly"] else 0 for r in collected.select("anomaly").collect()],
                )
                logger.info("z_max      {}", format_ranking(without_g))
                ranking.update({f"z_{k}": v for k, v in without_g.items()})
            else:
                logger.warning(
                    "dataset.anomaly_col is null: the ranking was produced but not "
                    "scored. Set it to the label column to get AUC-PR and ROC AUC."
                )

            quantiles = collected.approxQuantile(
                "score_max", [0.5, 0.9, 0.99, 0.999], 0.005
            )
            z_quantiles = collected.approxQuantile("z_max", [0.5, 0.9, 0.99], 0.005)

            logger.info(
                "score_max  p50 {:.2f}  p90 {:.2f}  p99 {:.2f}  p999 {:.2f}",
                *quantiles,
            )
            logger.info("Top {} by score_max:", top_n)
            for row in collected.orderBy(F.desc("score_max")).limit(top_n).collect():
                logger.info(
                    "  {} id={} score={:.1f} z={:.2f} residual={:.4f} "
                    "sigma={:.4f} g={:.4f} sim={:.5f} n={}",
                    row["date"],
                    row["id"],
                    row["score_max"],
                    row["z_max"],
                    row["residual_max"],
                    row["sigma_mean"],
                    row["expected_residual"],
                    row["best_sim"],
                    row["n_eff"],
                )

            write_csv(collected.orderBy(F.desc("score_max")), out_path)

            summary = {
                **ranking,
                "n_scored": n,
                "n_days": len(dates),
                "shrinkage": shrinkage,
                "score_p50": quantiles[0],
                "score_p90": quantiles[1],
                "score_p99": quantiles[2],
                "score_p999": quantiles[3],
                "z_p50": z_quantiles[0],
                "z_p90": z_quantiles[1],
                "z_p99": z_quantiles[2],
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            tracking.log_metrics(summary)
            mlflow.log_artifact(str(summary_path))
    finally:
        spark.stop()


if __name__ == "__main__":
    app()
