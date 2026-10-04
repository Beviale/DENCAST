"""Fit on train, apply to all three splits, then check what the result still holds."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from loguru import logger
import numpy as np
import pandas as pd

PARTS = ("train", "validation", "test")


class Processor:
    """Fit the transforms on train, apply them to every split, report the result."""

    def __init__(
        self,
        split_dir: Path,
        out_dir: Path,
        name: str,
        max_levels: int = 5,
        categorical: Optional[Sequence[str]] = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.out_dir = Path(out_dir)
        self.name = name
        self.max_levels = max_levels
        self.given_categorical = None if categorical is None else list(categorical)
        self.numeric: list[str] = []
        self.categorical: list[str] = []
        self.mean: pd.Series = pd.Series(dtype="float64")
        self.std: pd.Series = pd.Series(dtype="float64")
        self.median: pd.Series = pd.Series(dtype="float64")
        self.mode: pd.Series = pd.Series(dtype="float64")

    # ------------------------------------------------------------------- paths

    def in_path(self, part: str) -> Path:
        return self.split_dir / f"{self.name}_{part}.parquet"

    def out_path(self, part: str) -> Path:
        return self.out_dir / f"{self.name}_{part}.parquet"

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        """Everything that is not a label. Labels are carried through untouched."""
        return [c for c in df.columns if not c.startswith("is_")]

    # --------------------------------------------------------------------- fit

    def fit(self, train: pd.DataFrame) -> None:
        feats = self.feature_columns(train)
        levels = train[feats].nunique()
        guess = [c for c in feats if levels[c] <= self.max_levels]
        logger.info("fitted on {:,} training rows", len(train))

        if self.given_categorical is None:
            self.categorical = guess
            logger.warning("  categorical columns inferred: at most {} distinct values "
                        "in training", self.max_levels)
        else:
            absent = [c for c in self.given_categorical if c not in feats]
            present = [c for c in self.given_categorical if c in feats]
            if absent:
                logger.warning(
                    f"{len(absent)} of the {len(self.given_categorical)} columns "
                    f"named as categorical are not in the data: {', '.join(absent)}"
                )
            self.categorical = present
            logger.info("  categorical columns taken as given: {} named",
                        len(self.categorical))
            
        self.numeric = [c for c in feats if c not in set(self.categorical)]
        logger.info("  {} numeric, {} categorical",
                    len(self.numeric), len(self.categorical))
        
        self.mean = train[self.numeric].mean()
        self.std = train[self.numeric].std()
        self.median = train[self.numeric].median()

        flat = self.std.index[self.std <= 1e-12].tolist()
        if flat:
            logger.warning("  {} numeric columns are constant in training and are "
                           "centred but not scaled: {}", len(flat), ", ".join(flat))
            self.std = self.std.mask(self.std <= 1e-12, 1.0)

        modes = {}
        for c in self.categorical:
            m = train[c].mode(dropna=True)
            if len(m) == 0:
                logger.warning("  {} is entirely missing in training; it has no "
                               "mode and will not be imputed", c)
                continue
            if len(m) > 1:
                logger.warning("  {} has {} tied modes {}; the first is used",
                               c, len(m), list(m)[:4])
            modes[c] = m.iloc[0]
        self.mode = pd.Series(modes, dtype="float64")
        logger.info("  median fitted for {} numeric columns, mode for {} categorical",
                    len(self.median), len(self.mode))

    # --------------------------------------------------------------- transform

    def transform(self, df: pd.DataFrame, part: str) -> pd.DataFrame:
        out = df.copy()

        holes = out[self.numeric + self.categorical].isna()
        rows = int(holes.any(axis=1).sum())
        cells = int(holes.to_numpy().sum())
        if rows == 0:
            logger.info("{:<11} nothing to impute: 0 rows, 0 cells missing", part)
        else:
            logger.warning("{:<11} imputing {:,} rows ({:.4%} of the split), "
                           "{:,} cells", part, rows, rows / len(out), cells)
            per_col = holes.sum()
            for c, k in per_col[per_col > 0].items():
                how = "median" if c in self.numeric else "mode"
                fill = self.median.get(c) if c in self.numeric else self.mode.get(c)
                logger.warning("            {:<9} {:>8,} cells <- {} {} (from train)",
                               c, int(k), how, "n/a" if fill is None else f"{fill:g}")
            out[self.numeric] = out[self.numeric].fillna(self.median)
            if len(self.mode):
                out[self.categorical] = out[self.categorical].fillna(self.mode)

        out[self.numeric] = (out[self.numeric] - self.mean) / self.std
        return out

    # ------------------------------------------------------------------- check

    def check(self, parts: dict[str, pd.DataFrame]) -> None:
        """What survived. Reported whether or not anything did."""
        logger.info("final check")
        for part in PARTS:
            df = parts[part]
            feats = self.feature_columns(df)
        
            digest = pd.util.hash_pandas_object(df, index=True)
            dup = int(digest.duplicated(keep="first").sum())
            levels = df[feats].nunique()
            flat = levels.index[levels <= 1].tolist()
            nulls = int(df[feats].isna().to_numpy().sum())

            say = logger.success if not (dup or flat or nulls) else logger.warning
            say("  {:<11} {:>9,} rows x {} features   duplicate rows {:,}   "
                "constant columns {}   nulls {:,}",
                part, len(df), len(feats), dup, len(flat), nulls)
            if flat:
                say("              constant: {}", ", ".join(flat))
            if dup:
                say("              {:,} timestamps carry more than one row",
                    int(df.index.duplicated().sum()))

            by_feat = pd.util.hash_pandas_object(df[feats], index=False)
            for col in [c for c in df.columns if c.startswith("is_")]:
                votes = pd.DataFrame({"h": by_feat.to_numpy(), "y": df[col].to_numpy()})
                spread = votes.groupby("h")["y"].nunique()
                clashing = spread.index[spread > 1]
                if len(clashing) == 0:
                    continue
                n = int(votes["h"].isin(clashing).sum())
                logger.error("              {:,} rows in {:,} groups share every "
                             "feature but disagree on `{}`: no model can satisfy "
                             "both", n, len(clashing), col)

    # --------------------------------------------------------------------- run

    def run(self, overwrite: bool = False) -> dict[str, Path]:
        existing = [p for p in PARTS if self.out_path(p).exists()]
        if existing and not overwrite:
            logger.info("{} already exist, nothing to do",
                        ", ".join(self.out_path(p).name for p in existing))
            return {p: self.out_path(p) for p in PARTS}

        raw = {p: pd.read_parquet(self.in_path(p)) for p in PARTS}
        for p in PARTS:
            logger.info("{:<11} {:>9,} rows x {} columns from {}",
                        p, len(raw[p]), raw[p].shape[1], self.in_path(p).name)
        self.fit(raw["train"])

        done = {p: self.transform(raw[p], p) for p in PARTS}
        self.check(done)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for part in PARTS:
            tmp = self.out_path(part).with_suffix(".parquet.partial")
            done[part].to_parquet(tmp, compression="snappy")
            tmp.replace(self.out_path(part))
            written[part] = self.out_path(part)
            logger.success("  {}", self.out_path(part))
        return written


__all__ = ["Processor", "PARTS"]
