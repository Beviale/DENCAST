"""Metrics from a directory of scores, for any detector that can produce one."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger
import numpy as np
import pandas as pd

PARTS = ("train", "validation", "test")
COLUMNS = ("datetime", "score", "label")


def segments(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end of every run of consecutive positives, as half-open bounds."""
    edge = np.diff(np.r_[0, (y != 0).astype(int), 0])
    return np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)


def metrics(y: np.ndarray, score: np.ndarray,
            threshold: Optional[float] = None) -> dict:
    """Threshold-free scores, plus the ones that need a cut-off."""
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    usable = y.any() and not y.all()
    out = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)) if usable else None,
        "average_precision": float(average_precision_score(y, score)) if y.any()
        else None,
    }
    if threshold is None:
        grid = np.quantile(score, np.linspace(0.50, 0.9999, 400))
        f1s = [f1_score(y, score >= t, zero_division=0) for t in grid]
        threshold = float(grid[int(np.argmax(f1s))])
        out["threshold_source"] = "chosen here to maximise F1"
    else:
        out["threshold_source"] = "carried from validation"
    pred = (score >= threshold).astype("uint8")
    starts, ends = segments(y)
    caught = sum(1 for a, b in zip(starts, ends) if pred[a:b].any())
    out.update({
        "threshold": float(threshold),
        "flagged": int(pred.sum()),
        "flagged_share": float(pred.mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "segments": int(len(starts)),
        "segments_caught": int(caught),
        "segment_recall": float(caught / len(starts)) if len(starts) else None,
    })
    return out


def read_scores(path: Path) -> pd.DataFrame:
    """One part's scores, checked and in time order."""
    df = pd.read_parquet(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing {missing}; a scores file carries "
                         f"{', '.join(COLUMNS)}")
   
    df = df.sort_values("datetime", kind="stable")
    if df["score"].isna().any():
        raise ValueError(f"{path} holds {int(df['score'].isna().sum()):,} null scores")
    return df


def evaluate(scores_dir: Path | str,
             reports_dir: Path | str = Path("reports"),
             threshold: Optional[float] = None,
             threshold_from: str = "validation",
             name: Optional[str] = None) -> dict:
    """Score every part present and write one report."""
    scores_dir = Path(scores_dir)
    reports_dir = Path(reports_dir)
    meta_path = scores_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not meta:
        logger.warning("no meta.json in {}: the report will carry metrics only",
                       scores_dir)

    found = {p: scores_dir / f"{p}.parquet" for p in PARTS
             if (scores_dir / f"{p}.parquet").exists()}
    if not found:
        raise FileNotFoundError(f"no <part>.parquet in {scores_dir}")
    data = {p: read_scores(f) for p, f in found.items()}
    logger.info("{}: {}", scores_dir,
                ", ".join(f"{p} {len(d):,} windows" for p, d in data.items()))

    if threshold is None:
        threshold = (meta.get("model") or {}).get("threshold")
    if threshold is None and threshold_from in data:
        y = data[threshold_from]["label"].to_numpy()
        if y.any() and not y.all():
            threshold = metrics(y, data[threshold_from]["score"].to_numpy())["threshold"]
            logger.info("no threshold given: taken from {} at {:.4f}",
                        threshold_from, threshold)

    out = {}
    for p, d in data.items():
        y = d["label"].to_numpy()
        s = d["score"].to_numpy(dtype="float64")
        
        out[p] = metrics(y, s, threshold=threshold)
        m = out[p]
        if m["roc_auc"] is None:
            logger.info("  {:<11} {:,} windows, {} positive: ranking metrics need "
                        "both classes", p, m["rows"], m["positives"])
        else:
            logger.success("  {:<11} ROC-AUC {:.4f}  AP {:.4f}  F1 {:.4f}  "
                           "precision {:.4f}  recall {:.4f}  segments {}/{}",
                           p, m["roc_auc"], m["average_precision"], m["f1"],
                           m["precision"], m["recall"], m["segments_caught"],
                           m["segments"])

    report = {
        "model": meta.get("model", {}),
        "dataset": meta.get("dataset", {}),
        "metrics": {**{k: v for k, v in (meta.get("metrics") or {}).items()}, **out},
        "run": {
            "scores_dir": str(scores_dir),
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = name or scores_dir.name
    path = reports_dir / f"{stem}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.success("report written to {}", path)
    return report


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Metrics from a directory of per-window scores.")
    ap.add_argument("scores_dir", type=Path,
                    help="directory holding <part>.parquet and meta.json")
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--threshold", type=float, default=None,
                    help="overrides the one in meta.json")
    ap.add_argument("--threshold-from", default="validation",
                    help="part whose best-F1 cut-off is carried, when none is given")
    ap.add_argument("--name", default=None, help="report stem; defaults to the "
                                                 "scores directory name")
    a = ap.parse_args()
    evaluate(a.scores_dir, a.reports_dir, a.threshold, a.threshold_from, a.name)


if __name__ == "__main__":
    main()


__all__ = ["evaluate", "metrics", "segments", "read_scores", "PARTS", "COLUMNS"]
