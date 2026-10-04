"""ESA Anomaly Dataset, Mission 2: 100 independent channels into one table.

The dataset does not ship a table. Each telemetry channel is its own pickled
DataFrame with its own irregular time index, and the sampling rates differ by more
than three orders of magnitude -- `channel_1` carries 7.4 million samples at 18
seconds while `channel_100` carries 1,703 at roughly eighteen hours. There is no
rectangular view until one is built.

So the values are put on a **common 18-second grid with zero-order hold**, which is
what the dataset's own authors do: a channel keeps its last known value until it
reports a new one. The consequence is worth stating rather than hiding -- a slow
channel contributes a step function, constant for thousands of consecutive rows,
and any statistic computed over a window will reflect that.

**The labels are intervals, not a column.** `labels.csv` gives `(ID, Channel,
StartTime, EndTime)` and `anomaly_types.csv` says whether each ID is an `Anomaly`
or a `Rare Event`. So an annotation applies to *specific channels over a specific
stretch of time*, and turning it into a per-row label is work this class does
rather than something the dataset hands over.

**`Rare Event` is not an anomaly.** 613 of the 644 annotations are rare but
nominal behaviour, annotated precisely so they would not be mistaken for faults.
They are kept as a distinct level rather than folded into the positives: the
per-channel label is ternary (0 nominal, 1 anomaly, 2 rare event) so the choice of
what counts as a positive stays with whoever reads the table.

**Why two passes.** The window holds 3.7 million rows and the table is 202 columns
wide, which is 1.8 GB once -- and building it the obvious way costs that three
times over, in the columns, the frame that wraps them and the copy concatenation
makes. A first attempt was killed at the fortieth channel. So the first pass
resamples each channel and parks it on disk as a single-column `.npy`, holding one
raw channel at a time; the second reads those back through memory maps a block of
rows at a time and hands each block to the writer. Peak memory is then one raw
channel, around 120 MB, rather than a multiple of the finished table.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Iterator, Optional, Sequence

from loguru import logger
import numpy as np
import pandas as pd

from dencast.data.preprocess.preprocess import Preprocessor

NOMINAL, ANOMALY, RARE_EVENT = 0, 1, 2


class EsaPreprocessor(Preprocessor):
    """Mission 2 of the ESA Anomaly Dataset, windowed and put on one grid.

    The default window spans the whole recording, January 2000 to the end of June
    2003. The end is exclusive, so `2003-07-01` means "through 30 June 2003". Any
    channel reporting outside it is flagged rather than silently dropped.

    Narrower windows are cut from the finished table by `truncate_esa`, not by
    running this again: the expensive part is resampling 100 channels of 7.4
    million points each, and a date range does not change any of it.
    """

    def __init__(
        self,
        source: Path = Path("data/raw/esa/ESA-Mission2"),
        out_dir: Path = Path("data/interim/esa"),
        name: str = "full_esa",
        start: str = "2000-01-01",
        end: str = "2003-07-01",
        rule: str = "18s",
        channels: Optional[Sequence[str]] = None,
        target_only: bool = False,
        block_rows: int = 400_000,
    ) -> None:
        super().__init__(source, out_dir, name, start, end)
        self.rule = rule
        self.target_only = target_only
        # Before a channel first reports there is nothing to hold forward, so the
        # grid starts with NaN and keeps it. Carrying the first observed value
        # backwards would be inventing readings: the quantity existed, but nobody
        # measured it, and a table that cannot tell the two apart is worse than one
        # with holes in it. How many rows each channel lacks is logged, so the
        # decision about what to do with them is made downstream with the numbers
        # in hand.
        self._lead: dict[str, int] = {}
        # Rows per parquet row group. 400k by 202 columns is roughly 200 MB, which
        # keeps the second pass well inside what the machine has spare.
        self.block_rows = block_rows
        self._channels = list(channels) if channels is not None else None
        self._grid: Optional[pd.DatetimeIndex] = None
        self._annotations: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------- selection

    @property
    def channels(self) -> list[str]:
        """Channel names to load, in the order the index file lists them."""
        if self._channels is None:
            meta = pd.read_csv(self.source / "channels.csv")
            if self.target_only:
                # The benchmark scores a subset; restricting to it halves the
                # width without touching the rows.
                meta = meta.loc[meta["Target"] == "YES"]
            available = {p.stem for p in (self.source / "channels").glob("*.zip")}
            self._channels = [c for c in meta["Channel"] if c in available]
            missing = [c for c in meta["Channel"] if c not in available]
            if missing:
                logger.warning("{} channels listed but not on disk: {}",
                               len(missing), ", ".join(missing[:5]))
        return self._channels

    @property
    def grid(self) -> pd.DatetimeIndex:
        """The common time axis every channel is resampled onto."""
        if self._grid is None:
            self._grid = pd.date_range(self.start, self.end, freq=self.rule,
                                       inclusive="left", name="datetime")
        return self._grid

    # ------------------------------------------------------------ resampling

    def _resample(self, name: str, grid: pd.DatetimeIndex) -> np.ndarray:
        """One channel on `grid`, carrying its last reported value forward.

        `searchsorted` rather than a reindex-and-ffill: the union of a 7.4 million
        point index with the grid is itself larger than either, and building it
        per channel is most of what made the first attempt run out of memory.
        Here the grid is mapped straight onto the channel's own positions.
        """
        df = pd.read_pickle(self.source / "channels" / f"{name}.zip")
        s = df[name] if name in df.columns else df.iloc[:, 0]
        if s.dtype == object:
            # Ten channels are flagged Categorical; factorising keeps them in the
            # table as integer codes rather than dropping them.
            s = pd.Series(pd.factorize(s)[0], index=s.index, name=name)
        # Against the window, not against the grid's last point: the grid ends one
        # step before `end`, so comparing with it would flag every channel that
        # reports in the final interval -- which is all of them, and is not a
        # problem. What would be a problem is data outside the window the table
        # claims to cover.
        if s.index[0] < self.start or s.index[-1] >= self.end:
            logger.warning("{} reports from {} to {}, outside the window [{}, {})",
                           name, s.index[0], s.index[-1], self.start, self.end)
        pos = s.index.searchsorted(grid, side="right") - 1
        out = np.full(len(grid), np.nan, dtype=np.float32)
        seen = pos >= 0                       # before the channel's first report
        vals = s.to_numpy(dtype=np.float32, copy=False)
        out[seen] = vals[pos[seen]]
        self._lead[name] = int((~seen).sum())
        return out

    def _cache_channels(self, cache: Path) -> None:
        """First pass: every channel resampled onto the grid, parked on disk."""
        grid = self.grid
        logger.info("grid: {:,} rows at {} from {} to {} (exclusive)",
                    len(grid), self.rule, self.start, self.end)
        for i, name in enumerate(self.channels, 1):
            np.save(cache / f"{name}.npy", self._resample(name, grid))
            if i % 20 == 0 or i == len(self.channels):
                logger.info("  resampled {}/{} channels", i, len(self.channels))
        late = {k: v for k, v in self._lead.items() if v}
        if late:
            step = pd.Timedelta(self.rule).total_seconds()
            total = sum(late.values())
            logger.info("{} channels report for the first time after the window "
                        "opens; those rows stay NaN ({:,} in all)", len(late), total)
            for name, n in sorted(late.items(), key=lambda kv: -kv[1])[:5]:
                logger.info("    {:<14} {:>9,} NaN rows ({:.1f} days)",
                            name, n, n * step / 86400)

    # ----------------------------------------------------------------- values

    def load_values(self) -> pd.DataFrame:
        """Every channel on the common grid, one column each, all at once.

        Kept for the contract and for small channel subsets. The full run goes
        through `iter_blocks`, which never holds this.
        """
        grid = self.grid
        mat = np.empty((len(grid), len(self.channels)), dtype=np.float32)
        for i, name in enumerate(self.channels):
            mat[:, i] = self._resample(name, grid)
        return pd.DataFrame(mat, index=grid, columns=self.channels, copy=False)

    # ----------------------------------------------------------------- labels

    @property
    def annotations(self) -> pd.DataFrame:
        """Annotation intervals that touch the window, with their category."""
        if self._annotations is None:
            lab = pd.read_csv(self.source / "labels.csv",
                              parse_dates=["StartTime", "EndTime"])
            kinds = pd.read_csv(self.source / "anomaly_types.csv")[["ID", "Category"]]
            ann = lab.merge(kinds, on="ID", how="left")
            # The label timestamps carry a UTC offset and the channel index does
            # not. Comparing the two raises; dropping the zone is correct here
            # because the series is in UTC throughout, not because it is noise.
            for col in ("StartTime", "EndTime"):
                if isinstance(ann[col].dtype, pd.DatetimeTZDtype):
                    ann[col] = ann[col].dt.tz_convert(None)
            ann = ann[ann["Channel"].isin(set(self.channels))]
            ann = ann[(ann["EndTime"] >= self.grid[0]) & (ann["StartTime"] <= self.grid[-1])]
            logger.info("{} annotation intervals touch the window "
                        "({} anomalies, {} rare events)", len(ann),
                        int((ann["Category"] == "Anomaly").sum()),
                        int((ann["Category"] == "Rare Event").sum()))
            self._annotations = ann
        return self._annotations

    def load_labels(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Per-channel ternary labels plus two global flags, on `index`.

        Works on the whole grid or on one block of it, so the second pass can ask
        for a block's labels without materialising the rest.
        """
        cols = {c: np.zeros(len(index), dtype=np.uint8) for c in self.channels}
        # Anomalies are stamped after rare events so an overlap resolves to the
        # stronger claim: a stretch annotated both ways is an anomaly.
        for category, value in (("Rare Event", RARE_EVENT), ("Anomaly", ANOMALY)):
            sub = self.annotations[self.annotations["Category"] == category]
            for channel, s, e in zip(sub["Channel"], sub["StartTime"], sub["EndTime"]):
                lo = index.searchsorted(s, side="left")
                hi = index.searchsorted(e, side="right")
                if hi > lo:
                    cols[channel][lo:hi] = value

        labels = pd.DataFrame({f"label_{c}": v for c, v in cols.items()}, index=index)
        stacked = np.stack(list(cols.values()))
        labels["is_anomaly"] = (stacked == ANOMALY).any(axis=0).astype(np.uint8)
        labels["is_rare_event"] = (stacked == RARE_EVENT).any(axis=0).astype(np.uint8)
        return labels

    # ----------------------------------------------------------------- blocks

    def iter_blocks(self) -> Iterator[pd.DataFrame]:
        """Two passes: cache each channel on disk, then assemble row blocks."""
        grid = self.grid
        cache = Path(tempfile.mkdtemp(prefix="esa_resampled_"))
        logger.info("caching resampled channels under {}", cache)
        try:
            self._cache_channels(cache)
            maps = {c: np.load(cache / f"{c}.npy", mmap_mode="r") for c in self.channels}
            n_anom = n_rare = 0
            for lo in range(0, len(grid), self.block_rows):
                hi = min(lo + self.block_rows, len(grid))
                idx = grid[lo:hi]
                values = pd.DataFrame(
                    {c: np.asarray(m[lo:hi]) for c, m in maps.items()}, index=idx)
                labels = self.load_labels(idx)
                self.check_alignment(values, labels)
                n_anom += int(labels["is_anomaly"].sum())
                n_rare += int(labels["is_rare_event"].sum())
                yield pd.concat([values, labels], axis=1)
            logger.info("rows flagged: {:,} anomaly ({:.4%}), {:,} rare event ({:.4%})",
                        n_anom, n_anom / len(grid), n_rare, n_rare / len(grid))
        finally:
            shutil.rmtree(cache, ignore_errors=True)


def truncate_esa(
    src: Path = Path("data/interim/esa/full_esa.parquet"),
    dst: Path = Path("data/interim/esa/sampled_esa.parquet"),
    start: str = "2000-01-01",
    end: str = "2002-02-01",
) -> Path:
    """Cut a date window out of the finished table, one row group at a time.

    Reading the whole table to slice it would cost the memory the two-pass build
    was written to avoid, so the row groups are streamed: each is filtered on the
    index and appended, and only the groups that overlap the window are read at
    all.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    f = pq.ParquetFile(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".parquet.partial")
    writer = None
    rows = 0
    try:
        for g in range(f.metadata.num_row_groups):
            block = f.read_row_group(g).to_pandas()
            block = block.loc[(block.index >= lo) & (block.index < hi)]
            if block.empty:
                continue
            table = pa.Table.from_pandas(block, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="snappy")
            writer.write_table(table)
            rows += len(block)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"no rows of {src} fall in [{start}, {end})")
    tmp.replace(dst)
    logger.success("{}: {:,} rows from {} to {} (exclusive)  ({:.2f} GB on disk)",
                   dst.name, rows, start, end, dst.stat().st_size / 2**30)
    return dst


def main() -> None:
    EsaPreprocessor().run(overwrite=True)
    truncate_esa()


if __name__ == "__main__":
    main()


__all__ = ["EsaPreprocessor", "truncate_esa", "NOMINAL", "ANOMALY", "RARE_EVENT"]
