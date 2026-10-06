"""Score a split directory with a saved K-Means model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger

from dencast.modeling import windowing as W
from dencast.modeling.train_k_means import kmeans_config


def score_k_means(
    model_dir: Path | str,
    split_dir: Optional[Path | str] = None,
    scores_dir: Path | str = Path("reports/scores"),
    parts: Sequence[str] = W.PARTS,
    workers: Optional[int] = None,
) -> Path:
    """Write one parquet of scores per part, plus the meta the evaluator reads."""
    from pyspark.sql import functions as F

    model_dir = Path(model_dir)
    contract = json.loads((model_dir / "contract.json").read_text(encoding="utf-8"))
    win = contract["windowing"]
    workers = workers if workers is not None else kmeans_config().get("workers", 4)

    split_dir = Path(split_dir) if split_dir is not None \
        else Path(contract["dataset"]["split_dir"])
    dataset, files = W.discover(split_dir)
    if dataset != contract["dataset"]["name"]:
        logger.warning("the model was trained on `{}` and this split holds `{}`; "
                       "scoring anyway because the feature space is what has to "
                       "match", contract["dataset"]["name"], dataset)
    files = {p: files[p] for p in parts}
    logger.info("scoring {} with {} ({} s windows, {} features)",
                split_dir, model_dir, win["window_seconds"], len(win["features"]))

    spark = W.session(f"kmeans-score-{dataset}", workers)
    try:
        prep = W.prepare(spark, files, win["categorical"], win["window_seconds"],
                         parts, features=win["features"])
        centroids = contract["model"]["centroids"]
        if len(centroids) != contract["model"]["k_selected"]:
            raise ValueError(
                f"the contract carries {len(centroids)} centroids but says "
                f"k={contract['model']['k_selected']}")

        label, features = prep["label"], prep["features"]
        scores_dir = Path(scores_dir)
        scores_dir.mkdir(parents=True, exist_ok=True)
        written, counts = {}, {}
        for p in parts:
            out = (prep["vec"][p]
                   .select(F.col(W.INDEX).alias("datetime"),
                           W.scores(centroids, prep["vec"][p], features)
                           .alias("score"),
                           F.col(label).cast("int").alias("label"))
                   .orderBy("datetime"))
           
            frame = out.toPandas()
            path = scores_dir / f"{p}.parquet"
            frame.to_parquet(path, index=False)
            written[p] = str(path)
            counts[p] = len(frame)
            logger.info("  {:<11} {:>9,} rows -> {:>7,} windows, {:>6,} positive "
                        "({:.2%}) -> {}", p, prep["rows"][p], len(frame),
                        int(frame["label"].sum()), frame["label"].mean(), path.name)

        meta = {
            "model": contract["model"],
            "dataset": {**contract["dataset"],
                        "split_dir": str(split_dir),
                        "files": {p: str(files[p]) for p in parts},
                        "rows": {p: int(prep["rows"][p]) for p in parts},
                        "windows": {p: int(counts[p]) for p in parts}},
            "metrics": contract.get("metrics", {}),
            "run": {
                "model_dir": str(model_dir),
                "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }
        (scores_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                              encoding="utf-8")
    finally:
        spark.stop()

    logger.success("scores written to {}", scores_dir)
    return scores_dir


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Score a split directory with a saved K-Means model.")
    ap.add_argument("model_dir", type=Path, help="directory holding contract.json")
    ap.add_argument("--split-dir", type=Path, default=None,
                    help="data to score; defaults to the one in the contract")
    ap.add_argument("--scores-dir", type=Path, default=None,
                    help="output; defaults to reports/scores/<model name>")
    ap.add_argument("--parts", nargs="+", default=list(W.PARTS))
    ap.add_argument("--workers", type=int, default=None, help="local Spark cores")
    a = ap.parse_args()
    out = a.scores_dir or Path("reports/scores") / a.model_dir.name
    score_k_means(a.model_dir, a.split_dir, out, a.parts, a.workers)


if __name__ == "__main__":
    main()


__all__ = ["score_k_means"]
