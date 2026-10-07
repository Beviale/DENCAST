"""
Fit K-Means on a split's training part, choose k on validation, and finally save the model and
test on the best validation.

The method is one-class: K-Means is fitted on data believed to be normal and a window
is scored by its distance to the nearest centroid. Nothing about the clustering uses a
label; labels enter only to choose 'k' and to place the threshold.

**The rows are aggregated into fixed time windows first.** 

Five stages here:

1. For each candidate 'k', fit on train and score validation. Train is attack-free on
   SWaT and HAI by construction, and on ESA because the processor removed the labelled
   rows, so this is a one-class fit.
2. Pick the 'k' with the best validation AveragePrecision -- threshold-free.
3. Refit the model on train+validation with the best k 
4. Compute the test metrics
5. Save the model


**The refit has a trap, and the default avoids it.** Validation may contain attacks.
Fitting a model of normality on data that includes them lets the anomalies pull a
centroid towards themselves, and anything similar at test time then sits close to that
centroid and scores as normal. So the refit uses the normal windows of validation by
default.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from dencast.modeling import windowing as W
from dencast.modeling.evaluate import metrics
from dencast.utils import Params, PARAMS_PATH, declared_categorical

from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.clustering import KMeans


DEFAULT_K = (2, 4, 8, 16, 32, 64)
DEFAULT_WINDOW = 60


def kmeans_config() -> dict:
    """The 'kmeans' section of params.yaml, or an empty dict if it cannot be read."""
    try:
        section = Params.load(PARAMS_PATH).get("kmeans")
        return section.to_dict() if hasattr(section, "to_dict") else (section or {})
    except Exception as exc:
        logger.warning("params.yaml unreadable ({}): falling back to the built-in "
                       "defaults", exc)
        return {}


def get_distributed_scorer(centroids):
    """Create a Pandas UDF wuth the current centroids."""
    centroids_arr = np.array(centroids)
    
    @F.pandas_udf(DoubleType())
    def _scorer(batch: pd.Series) -> pd.Series:
        X_batch = np.stack(batch.values)
        
        dists = cdist(X_batch, centroids_arr, metric="euclidean")
        
        return pd.Series(dists.min(axis=1))
        
    return _scorer

def train_k_means(
    split_dir: Path | str,
    models_dir: Path | str = Path("models"),
    reports_dir: Path | str = Path("reports"),
    k_values: Optional[Sequence[int]] = None,
    window_seconds: Optional[int] = None,
    seed: Optional[int] = None,
    max_iter: Optional[int] = None,
    contaminated_refit: Optional[bool] = None,
    workers: Optional[int] = None,
) -> Path:
    """Train, select k, refit, save. Returns the model directory."""
    cfg = kmeans_config()
    k_values = k_values if k_values is not None else cfg.get("k_values", DEFAULT_K)
    seed = seed if seed is not None else cfg.get("seed", 42)
    max_iter = max_iter if max_iter is not None else cfg.get("max_iter", 50)
    workers = workers if workers is not None else cfg.get("workers", 4)
    if contaminated_refit is None:
        contaminated_refit = bool(cfg.get("contaminated_refit", False))


    split_dir, models_dir = Path(split_dir), Path(models_dir)
    dataset, files = W.discover(split_dir)
    if window_seconds is None:
        by_dataset = cfg.get("window_seconds") or {}
        window_seconds = by_dataset.get(dataset, DEFAULT_WINDOW)
        source = "params.yaml" if dataset in by_dataset else "built-in default"
    else:
        source = "the command line"
    logger.info("dataset {} from {}", dataset, split_dir)
    logger.info("k values {} | window {} s (from {})",
                list(k_values), window_seconds, source)

    spark = W.session(f"kmeans-train-{dataset}", workers)
    try:
        parts = ("train", "validation")
        prep = W.prepare(spark, files, declared_categorical(dataset),
                         window_seconds, parts)
        vec, features, label = prep["vec"], prep["features"], prep["label"]

        y = {p: np.asarray([r[label] for r in vec[p].select(label).collect()],
                           dtype="uint8") for p in parts}
        for p in parts:
            logger.info("  {:<11} {:>9,} rows -> {:>7,} windows, {:>6,} positive "
                        "({:.2%})", p, prep["rows"][p], len(y[p]),
                        int(y[p].sum()), y[p].mean())
        if y["train"].any():
            logger.warning("  the training part is not clean: a one-class fit on it "
                           "is fitting normality to {:,} positive windows",
                           int(y["train"].sum()))

        search = []
        vec_val = vec["validation"].withColumn("features_arr", vector_to_array("features"))
        for k in k_values:
            model = KMeans(k=k, seed=seed, maxIter=max_iter,
                           featuresCol="features").fit(vec["train"])
            centroids = model.clusterCenters()
            s_df = vec_val.select(
                get_distributed_scorer(centroids)(F.col("features_arr")).alias("_s")
            )
            s = np.asarray([r["_s"] for r in s_df.collect()], dtype="float64")
            m = metrics(y["validation"], s)
            search.append({"k": int(k), **m})
            logger.info("  k={:<3} validation AP {:.4f} ROC-AUC {:.4f}", k, m["average_precision"], m["roc_auc"])

        best = max(search, key=lambda r: r["average_precision"])
        k_star, best_threshold, best_average_precision = best["k"], best["threshold"], best["average_precision"]
        logger.success("selected k={} on validation AveragePrecision {:.4f}, threshold selected={:.4f}",
                       k_star, best_average_precision, best_threshold)

        if contaminated_refit:
            refit = vec["train"].unionByName(vec["validation"])
            note = "train + all of validation, attacks included"
        else:
            refit = vec["train"].unionByName(vec["validation"].filter(f"{label} = 0"))
            note = "train + the normal windows of validation"
        n_refit = refit.count()
        logger.info("refitting k={} on {:,} windows ({})", k_star, n_refit, note)

        vec_test = vec["test"].withColumn("features_arr", vector_to_array("features"))
        final = KMeans(k=k_star, seed=seed, maxIter=max_iter,
                       featuresCol="features").fit(refit)
        centroids = model.clusterCenters()
        s_df = vec_test.select(
            get_distributed_scorer(centroids)(F.col("features_arr")).alias("_s")
        )
        s = np.asarray([r["_s"] for r in s_df.collect()], dtype="float64")
        best_validation_on_test_set = metrics(y["test"], s, best_threshold)
        logger.info("Best validation on test set:  AP {:.4f}, ROC-AUC {:.4f}, F1 {:.4f}", 
        best_validation_on_test_set["average_precision"], best_validation_on_test_set["roc_auc"], best_validation_on_test_set["f1"])

        contract_out = models_dir / f"kmeans_{split_dir.name}_{window_seconds}s"
        contract_out.mkdir(parents=True, exist_ok=True)
        test_out = reports_dir /f"kmeans_{split_dir.name}_{window_seconds}s"
        centroids = [[float(x) for x in c] for c in final.clusterCenters()]

        contract = {
            "model": {
                "name": "KMeans",
                "library": "pyspark.ml.clustering",
                "spark_version": spark.version,
                "score": "euclidean distance to the nearest centroid",
                "k_candidates": [int(k) for k in k_values],
                "k_selected": k_star,
                "selected_by": "validation AveragePrecision",
                "seed": seed,
                "max_iter": max_iter,
                "refit_on": note,
                "refit_windows": int(n_refit),
                "centroids": centroids,
            },
            "dataset": {
                "name": dataset,
                "split_dir": str(split_dir),
                "window_seconds": int(window_seconds),
                "source_columns": len(prep["columns"]),
                "features": len(features),
                "feature_columns": features,
                "label_column": label,
            },
            "metrics": {"validation_search": search, "validation_best": best, "best_validation_on_test_set":best_validation_on_test_set},
            "run": {
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }
        (contract_out / "contract.json").write_text(json.dumps(contract, indent=2),
                                           encoding="utf-8")

        test_json_metrics = {
            "model": {
                "name": "KMeans",
                "library": "pyspark.ml.clustering",
                "spark_version": spark.version,
                "score": "euclidean distance to the nearest centroid",
                "k_candidates": [int(k) for k in k_values],
                "k_selected": k_star,
                "selected_by": "validation AveragePrecision",
                "seed": seed,
                "max_iter": max_iter,
                "refit_on": note,
                "refit_windows": int(n_refit),
            },
            "dataset": {
                "name": dataset,
                "split_dir": str(split_dir),
                "window_seconds": int(window_seconds),
                "source_columns": len(prep["columns"]),
                "features": len(features),
                "feature_columns": features,
                "label_column": label,
            },
            "metrics": {"best_validation_on_test_set":best_validation_on_test_set},
            "run": {
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        }
        (test_out / "best_validation_on_test_set.json").write_text(json.dumps(test_json_metrics, indent=2),
                                           encoding="utf-8")
    finally:
        spark.stop()

    logger.success("model and contract written to {}", contract_out)
    logger.success("test metrics on best validation written to {}", test_out)
    return contract_out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Fit K-Means on a split's training part and save the model.")
    ap.add_argument("--split_dir", type=Path,
                    help="directory holding *_train/_validation/_test.parquet", default="data/processed/swat")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--window-seconds", type=int, default=None,
                    help="overrides params.yaml; 1 or 0 disables windowing")
    ap.add_argument("--k", type=int, nargs="+", default=None,
                    help="overrides kmeans.k_values in params.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--contaminated-refit", action="store_true", default=None,
                    help="refit on all of validation, attacks included")
    ap.add_argument("--workers", type=int, default=None, help="local Spark cores")
    a = ap.parse_args()
    train_k_means(a.split_dir, a.models_dir, a.reports_dir, a.k, a.window_seconds, a.seed,
                  contaminated_refit=a.contaminated_refit, workers=a.workers)


if __name__ == "__main__":
    main()


__all__ = ["train_k_means", "kmeans_config", "DEFAULT_K", "DEFAULT_WINDOW"]
