from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd

from dencast.data.split.split import Splitter, PARTS

LEAD_IN_END = "2000-02-28 06:22:30"
VALID_START = "2001-10-01"
TEST_START = "2002-04-03"
DIRTY = ("is_anomaly", "is_rare_event")


class EsaSplitter(Splitter):
    """Chronological, with the lead-in dropped and the training part cleaned."""

    def __init__(
        self,
        source: Path = Path("data/interim/esa/full_esa.parquet"),
        out_dir: Path = Path("data/interim/esa/split"),
        valid_start: str = VALID_START,
        test_start: str = TEST_START,
        name: str = "esa",
        clean_train: bool = True,
    ) -> None:
        super().__init__(source, out_dir, valid_start, test_start, name)
        self.clean_train = clean_train


    def cut(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        parts = super().cut(df)
        if not self.clean_train:
            train = parts["train"]
            for c in [c for c in DIRTY if c in train.columns]:
                k = int(train[c].to_numpy().sum())
                logger.warning("the training part keeps {:,} rows flagged `{}` "
                               "({:.2%}): the fit is contaminated by design", k, c,
                               k / len(train))
            return parts
        train = parts["train"]
        flags = [c for c in DIRTY if c in train.columns]
        dirty = np.zeros(len(train), dtype=bool)
        for c in flags:
            dirty |= train[c].to_numpy() != 0
        logger.info("cleaning the training part: dropping {:,} of {:,} rows "
                    "({:.2%}) flagged by {}", int(dirty.sum()), len(train),
                    dirty.mean(), " or ".join(flags))
        for c in flags:
            k = int(train[c].to_numpy().sum())
            logger.info("    {:<14} {:>8,} rows", c, k)
        parts["train"] = train.loc[~dirty]
        if parts["train"].empty:
            raise ValueError("the training part is empty once cleaned")
        return parts

    def check_segments(self, df: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> None:
        later = df.loc[df.index >= self.valid_start]
        for col in [c for c in df.columns if c.startswith("is_")]:
            whole = self.segments(later[col].to_numpy())
            pieces = sum(self.segments(parts[p][col].to_numpy())
                         for p in ("validation", "test"))
            if pieces != whole:
                raise ValueError(
                    f"the test cut falls inside a run of `{col}`: validation and "
                    f"test hold {pieces} runs where the stretch from "
                    f"{self.valid_start} holds {whole}"
                )
        logger.info("neither cut falls inside a labelled run")


def main() -> None:
    EsaSplitter().run(overwrite=True)


if __name__ == "__main__":
    main()


__all__ = ["EsaSplitter", "LEAD_IN_END", "VALID_START", "TEST_START"]
