from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

PARTS = ("train", "validation", "test")


class Processor:
    """Fit the transforms on train, apply them to every split, report the result."""

    def __init__(
        self,
        split_dir: Path,
        out_dir: Path,
        name: str,
        categorical: Sequence[str],
        pearson_max: float = 0.99,
        cramer_max: float = 0.99,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.out_dir = Path(out_dir)
        self.name = name
        if categorical is None:
            raise ValueError(
                "the categorical columns must be given!")
        self.given_categorical = list(categorical)
        # A pair above either threshold is one quantity reported twice. A copy is dropped. For numeric variables.
        self.pearson_max = pearson_max
        # A pair above either threshold is one quantity reported twice. A copy is dropped. For categorical variables.
        self.cramer_max = cramer_max
        self.dropped_numeric: list[tuple[str, str, float]] = []
        self.dropped_categorical: list[tuple[str, str, float]] = []
        self.constant: list[str] = []
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
        """Everything that is not a label."""
        return [c for c in df.columns
                if not c.startswith("is_") and not c.startswith("label_")]

    def prepare_train(self, train: pd.DataFrame) -> pd.DataFrame:
        """Last chance to change the training rows before anything is fitted."""
        return train

    # --------------------------------------------------------------------- fit

    def fit(self, train: pd.DataFrame) -> None:
        feats = self.feature_columns(train)
        logger.info("fitted on {:,} training rows", len(train))

        absent = [c for c in self.given_categorical if c not in feats]
        self.categorical = [c for c in self.given_categorical if c in feats]
        if absent:
            logger.warning(
                f"{len(absent)} of the {len(self.given_categorical)} columns named as "
                f"categorical are not in the data: {', '.join(absent)}"
            )
        logger.info("  categorical columns taken as given: {} named",
                    len(self.categorical))

        self.numeric = [c for c in feats if c not in set(self.categorical)]
        logger.info("  {} numeric, {} categorical",
                    len(self.numeric), len(self.categorical))
        
        self.drop_constant(train)
        self.prune(train)

        self.mean = train[self.numeric].mean()
        self.std = train[self.numeric].std()
        self.median = train[self.numeric].median()

    
        flat = self.std.index[self.std == 0].tolist()
        if flat:
            logger.warning("  {} numeric columns have zero spread in training yet "
                           "survived the constant filter; centred but not scaled: {}",
                           len(flat), ", ".join(flat))
            self.std = self.std.mask(self.std == 0, 1.0)

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

    def drop_constant(self, train: pd.DataFrame) -> None:
        feats = self.numeric + self.categorical
        levels = train[feats].nunique()
        self.constant = [c for c in feats if levels[c] <= 1]
        if not self.constant:
            logger.info("  no column is constant across the training split")
            return
        gone = set(self.constant)
        in_num = [c for c in self.numeric if c in gone]
        in_cat = [c for c in self.categorical if c in gone]
        self.numeric = [c for c in self.numeric if c not in gone]
        self.categorical = [c for c in self.categorical if c not in gone]
        logger.info("  constant across the whole training split: {} dropped "
                    "({} numeric, {} categorical)",
                    len(self.constant), len(in_num), len(in_cat))
        logger.info("    {}", ", ".join(self.constant))

    @staticmethod
    def cramer_v(a: pd.Series, b: pd.Series) -> float:
        table = pd.crosstab(a, b)
        if min(table.shape) < 2:
            return 0.0
        chi2 = chi2_contingency(table)[0]
        return float(np.sqrt(chi2 / (len(a) * (min(table.shape) - 1))))

    def prune(self, train: pd.DataFrame) -> None:
        before_num, before_cat = len(self.numeric), len(self.categorical)
        levels = train[self.numeric + self.categorical].nunique()


        def finest(cols: list[str]) -> list[str]:
            return sorted(cols, key=lambda c: (-int(levels[c]), cols.index(c)))

        corr = train[self.numeric].corr().abs() if self.numeric else None
        keep, dropped = [], []
        for c in finest(self.numeric):
            hit = next((k for k in keep if corr.at[c, k] > self.pearson_max), None)
            if hit is None:
                keep.append(c)
            else:
                dropped.append((c, hit, float(corr.at[c, hit])))
        kept = set(keep)
        self.numeric = [c for c in self.numeric if c in kept]
        self.dropped_numeric = dropped

        keep, dropped = [], []
        for c in finest(self.categorical):
            hit = None
            for k in keep:
                v = self.cramer_v(train[c], train[k])
                if v > self.cramer_max:
                    hit = (k, v)
                    break
            if hit is None:
                keep.append(c)
            else:
                dropped.append((c, hit[0], hit[1]))
        kept = set(keep)
        self.categorical = [c for c in self.categorical if c in kept]
        self.dropped_categorical = dropped

        logger.info("  redundancy filter: Pearson > {:.3f}, Cramer's V > {:.3f}, "
                    "keeping the column with more distinct values",
                    self.pearson_max, self.cramer_max)
        logger.info("    numeric     {} -> {}  ({} dropped)", before_num,
                    len(self.numeric), len(self.dropped_numeric))
        for c, k, v in self.dropped_numeric:
            logger.info("      {} ({:,} levels) dropped, r = {:+.4f} with {} ({:,})",
                        c, int(levels[c]), v, k, int(levels[k]))
        logger.info("    categorical {} -> {}  ({} dropped)", before_cat,
                    len(self.categorical), len(self.dropped_categorical))
        for c, k, v in self.dropped_categorical:
            logger.info("      {} ({} levels) dropped, V = {:.4f} with {} ({})",
                        c, int(levels[c]), v, k, int(levels[k]))
        logger.info("    features    {} -> {}", before_num + before_cat,
                    len(self.numeric) + len(self.categorical))

    # --------------------------------------------------------------- transform

    def transform(self, df: pd.DataFrame, part: str) -> pd.DataFrame:
        gone = self.constant + [c for c, _, _ in
                                self.dropped_numeric + self.dropped_categorical]
        out = df.drop(columns=gone).copy()

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
        raw["train"] = self.prepare_train(raw["train"])
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
