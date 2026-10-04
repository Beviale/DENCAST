"""Dividing the days: train/test, and the days sampled inside train.

The paper shipped ten fixed lists of test days as text files. Measured, they
were independent random 10% samples rather than folds: any two shared about
nine days, and 36% of the dates appeared in more than one list. Nothing
recorded which days had been used to fit what. This module replaces them.

**The train/test split is chronological.** DENCAST predicts a day from the
`window_size` days before it, so a shuffled split would drop future days into
the training window of past ones. The leak is not hypothetical on this data --
a plant's output is strongly autocorrelated, so a later day in the window is
close to giving the answer away -- and it flatters every number downstream
without raising anything.

**Three slices, because choosing and reporting are different measurements.**
Train fits the model, valid scores the grid and picks the winner, test is
touched once at the end. Score the grid on test and the reported number is a
best-of-N maximum dressed up as an estimate -- the more configurations tried,
the more it overstates, and nothing is left to catch it. The middle slice is
what makes the final number mean what it says.

Valid carries the anomaly label, because the grid is scored on detection and
detection metrics need labels. That is a real concession and worth naming: the
search is supervised in its model *selection* even though every model it fits
is unsupervised. Train keeps no label at all, so nothing that fits can reach
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from typing import List

from loguru import logger
from pyspark.sql import DataFrame

from dencast.utils import Params

DATE_FORMAT = "%Y-%m-%d"


@dataclass(frozen=True)
class TrainValidTestDays:
    """The three day lists, each chronological and contiguous."""

    train: List[str]
    valid: List[str]
    test: List[str]

    def __repr__(self) -> str:
        def span(days: List[str]) -> str:
            return f"{len(days)} ({days[0]} .. {days[-1]})" if days else "0"

        return (
            f"TrainValidTestDays(train={span(self.train)}, "
            f"valid={span(self.valid)}, test={span(self.test)})"
        )


@dataclass(frozen=True)
class TrainTestDays:
    """The two day lists, both chronological."""

    train: List[str]
    test: List[str]

    def __repr__(self) -> str:
        def span(days: List[str]) -> str:
            return f"{len(days)} ({days[0]} .. {days[-1]})" if days else "0"

        return f"TrainTestDays(train={span(self.train)}, test={span(self.test)})"


def all_dates(df: DataFrame) -> List[str]:
    """Every distinct day in the data, in order."""
    days = sorted(r["date"] for r in df.select("date").distinct().collect())
    if not days:
        raise ValueError("The dataset has no dates")
    return days


def chronological_split(days: List[str], train_fraction: float) -> TrainTestDays:
    """Cut a chronological day list at `train_fraction`.

    The boundary falls between two days, never inside one: an object and the
    window it would be predicted from must not end up on opposite sides.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    cut = int(round(len(days) * train_fraction))
    cut = max(1, min(cut, len(days) - 1))
    return TrainTestDays(train=days[:cut], test=days[cut:])


def usable_days(days: List[str], first_day: str, window_size: int) -> List[str]:
    """Days with a complete training window behind them.

    The earliest days of a split cannot be evaluated fairly: their sliding
    window runs off the front, so they would be judged on a smaller training
    set than every other day.
    """
    earliest = datetime.strptime(first_day, DATE_FORMAT) + timedelta(
        days=window_size + 1
    )
    return [d for d in days if datetime.strptime(d, DATE_FORMAT) >= earliest]


def sample_days(days: List[str], n: int, seed: int) -> List[str]:
    """Draw n days at random, without replacement, and return them in order.

    Random rather than the first n: the first usable days of PV Italy all fall
    in one season, and selecting hyperparameters on winter alone would tune the
    model for half the year. The seed makes the draw reproducible, and changing
    it is how you ask whether a result was the luck of the draw.
    """
    if n > len(days):
        raise ValueError(
            f"Asked for {n} days but only {len(days)} are usable. Lower "
            "selection.n_days, or shorten evaluation.window_size so that more "
            "of the early days become usable."
        )
    return sorted(random.Random(seed).sample(days, n))


def split_days(df: DataFrame, params: Params) -> TrainTestDays:
    """The chronological train/test division of the whole dataset."""
    days = all_dates(df)
    split = chronological_split(days, params.split.train_fraction)
    logger.info(
        "{}  ({:.0%} / {:.0%} of {} days)",
        split,
        params.split.train_fraction,
        1 - params.split.train_fraction,
        len(days),
    )
    return split


def selection_days(train_days: List[str], params: Params) -> List[str]:
    """The days the hyperparameter search evaluates on, drawn from train."""
    usable = usable_days(train_days, train_days[0], params.evaluation.window_size)
    chosen = sample_days(usable, params.selection.n_days, params.selection.seed)
    logger.info(
        "Selection days: {} drawn from {} usable of {} train days ({} .. {})",
        len(chosen),
        len(usable),
        len(train_days),
        chosen[0],
        chosen[-1],
    )
    return chosen




def chronological_split_3(
    days: List[str], train_fraction: float, valid_fraction: float
) -> TrainValidTestDays:
    """Cut a chronological day list into train, valid and test.

    Both boundaries fall between days, never inside one, and every slice keeps
    at least one day so that a lopsided fraction fails loudly here instead of
    producing an empty split that some later stage reports as zero.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError(f"valid_fraction must be in [0, 1), got {valid_fraction}")
    if train_fraction + valid_fraction >= 1.0:
        raise ValueError(
            f"train_fraction + valid_fraction = {train_fraction + valid_fraction} "
            "leaves no days for test"
        )
    n = len(days)
    if n < 3:
        raise ValueError(f"need at least 3 days to make three splits, got {n}")

    cut1 = max(1, min(int(round(n * train_fraction)), n - 2))
    cut2 = max(cut1 + 1, min(int(round(n * (train_fraction + valid_fraction))), n - 1))
    return TrainValidTestDays(train=days[:cut1], valid=days[cut1:cut2], test=days[cut2:])


def split_days_3(df: DataFrame, params: Params) -> TrainValidTestDays:
    """The chronological train/valid/test division of the whole dataset."""
    days = all_dates(df)
    split = chronological_split_3(
        days, params.split.train_fraction, params.split.valid_fraction
    )
    logger.info(
        "{}  ({:.0%} / {:.0%} / {:.0%} of {} days)",
        split,
        params.split.train_fraction,
        params.split.valid_fraction,
        params.split.test_fraction,
        len(days),
    )
    return split


def evaluation_days(
    days: List[str], first_day: str, n: int, window_size: int, seed: int
) -> List[str]:
    """The days actually scored inside a split: usable, sampled, in order.

    `first_day` is the first day of the *whole* dataset, not of the split. A
    day at the start of valid still has the end of train behind it, and that
    window is legitimate -- it is the past. Passing the split's own first day
    would discard the first month of valid and of test for no reason.
    """
    usable = usable_days(days, first_day, window_size)
    if not usable:
        raise ValueError(
            f"No day in this split has a complete {window_size}-day window behind it"
        )
    return sample_days(usable, min(n, len(usable)), seed)
