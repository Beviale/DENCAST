
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyspark.sql import Column
from pyspark.sql import functions as F
import yaml

from dencast.config import PARAMS_PATH


def norm(a: Column) -> Column:
    """L2 norm of an array<double> column."""
    squares = F.transform(a, lambda x: x * x)
    return F.sqrt(F.aggregate(squares, F.lit(0.0), lambda acc, x: acc + x))


@dataclass
class DatasetParams:
    name: str
    source: str
    """Path to the source file, relative to the project root.

    A `.sql` pg_dump (the datasets of the paper) or a plain `.csv`. Keep it in
    data/external/: data/raw/ is a DVC output and gets overwritten.
    """

    id_col: str
    feature_cols: List[str] = field(default_factory=list)
    target_cols: List[str] = field(default_factory=list)

    columns: List[str] = field(default_factory=list)
    """Every column the model uses, with no target singled out.

    The anomaly task has no target to predict: an object is judged by how far
    each of its columns sits from its cluster, and a column withheld as a
    target would simply be a column not judged. Setting this replaces
    `feature_cols` and `target_cols` -- they become the whole list and empty
    respectively, so the vector the clustering is built on is all of it.

    `feature_cols`/`target_cols` remain for the regression pipeline, which the
    diagnostic scripts under scripts/ still use.
    """

    table: Optional[str] = None
    """Table to pull out of the dump. Required for a .sql source, ignored for CSV."""

    date_col: Optional[str] = None
    group_col: Optional[str] = None

    subsample_minutes: Optional[int] = None
    """Keep one reading every N minutes, dropping the rest.

    Applied while extracting, where the timestamp still has its time of day;
    by the time the pipeline sees it, `date_col` has been truncated to a date.

    On sub-hourly data this is not only about speed. Ten minutes apart, a
    turbine in steady wind reports the same thing twice: the readings are not
    independent observations, yet they enter a cluster as two members. The
    shrinkage and the leave-one-out statistics both assume independence, so
    the variance a cluster reports is biased low before any anomaly is
    scored. Thinning to one reading an hour removes redundancy that was
    distorting the score, and happens to cut the cost of the similarity join
    quadratically.
    """

    cyclical_cols: Dict[str, float] = field(default_factory=dict)
    """Feature columns that wrap around, mapped to the value of one full cycle.

    A compass bearing of 359 and one of 1 are two degrees apart, but as plain
    numbers they sit at opposite ends of the range -- so a model built on
    distances treats them as maximally different. Each column listed here is
    replaced by the sine and cosine of its angle, which restores the adjacency.

    The period is whatever value completes one turn *in the units the column
    actually uses*: 360 for degrees, 24 for hours, 1.0 for a column already
    normalised to [0, 1].
    """

    anomaly_col: Optional[str] = None
    """Column flagging an anomalous observation, or None when unlabeled.

    Anything truthy counts: a boolean, or a number greater than zero. When it
    is set the column is carried through to the processed data as `anomaly`
    and the supervised metrics are reported; when it is None they are skipped.
    """

    scale_features: bool = True

    def __post_init__(self) -> None:
        # `columns` wins when present: everything downstream reads feature_cols
        # and target_cols, so deriving them here means no other module has to
        # know which of the two styles the YAML used.
        if self.columns:
            self.feature_cols = list(self.columns)
            self.target_cols = []

    @property
    def m(self) -> int:
        """Number of descriptive attributes, after the cyclical expansion.

        Each cyclical column becomes two, so it adds one to the count.
        """
        return len(self.feature_cols) + len(self.cyclical_cols)

    @property
    def k(self) -> int:
        """Number of target attributes. k=1 single-target, k>1 multi-target."""
        return len(self.target_cols)


@dataclass
class LshParams:
    r: int = 11
    num_permutations: int = 20
    b: int = 10
    min_sim: float = 0.95
    use_targets_in_signature: bool = True


@dataclass
class ClusteringParams:
    min_pts: int = 5
    label_change_rate: float = 0.05
    max_iterations: int = 100


@dataclass
class SplitParams:
    """The train/test division, by fraction of days and chronological.

    Chronological rather than shuffled because DENCAST predicts a day from the
    days before it: a random split would place future days inside the training
    window of past ones, and every number downstream would be flattered by
    information the model could not have had.
    """

    train_fraction: float = 0.7

    valid_fraction: float = 0.15
    """The middle slice, on which the grid search is scored.

    It exists so that choosing a configuration and reporting it are not the
    same measurement. Score the grid on test and the winner is the point that
    happened to suit those days, with nothing left to check it against; the
    number would be a best-of-N maximum reported as if it were an estimate.
    Train fits, valid chooses, test reports once.
    """

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.valid_fraction


@dataclass
class ScoringParams:
    """The anomaly score: how a column deviation becomes one number.

    No calibration curve and no `g`. That machinery existed to say how large a
    residual was normal for a given assignment confidence, and there is no
    residual here -- nothing is predicted. The similarity enters the score
    directly instead, as a factor in the `sim` formula.
    """

    offset: float = 1.0
    """Midpoint of the sigmoid, in mean z-squared per column."""

    scale: float = 1.0
    """Width of the sigmoid, in the same units."""

    shrinkage: float = 2.0
    """Pseudo-count pulling each cluster variance toward the global one."""

    exclude_cols: List[str] = field(default_factory=list)
    """Columns to leave out of the deviation, though they stay in the routing.

    For identifiers and fixed geometry -- a plant id, its coordinates -- a
    large z means the object was routed to a cluster of other plants, which is
    a fact about the routing and not a fault.
    """


@dataclass
class SelectionParams:
    """The hyperparameter search, which runs inside the train split only.

    `grid` maps a dotted parameter path -- "lsh.min_sim", "clustering.min_pts"
    -- to the values to try. Cost is |combinations| x n_days model fits, so an
    extra value on any axis multiplies the run time rather than adding to it.
    """

    n_days: int = 5
    seed: int = 42
    metric: str = "average_precision"
    grid: Dict[str, List[Any]] = field(default_factory=dict)

    def combinations(self) -> List[Dict[str, Any]]:
        """Every point of the grid, as {dotted path: value}.

        Ordered so the run is reproducible: itertools.product over the keys in
        sorted order, which makes the same grid give the same sequence whatever
        order the YAML happened to list it in.
        """
        import itertools

        if not self.grid:
            return [{}]
        keys = sorted(self.grid)
        return [
            dict(zip(keys, values))
            for values in itertools.product(*(self.grid[k] for k in keys))
        ]


@dataclass
class EvaluationParams:
    window_size: int = 30
    seeds: List[int] = field(default_factory=lambda: [42])
    n_calibration_days: int = 5


@dataclass
class SparkParams:
    master: str = "local[*]"
    driver_memory: str = "4g"
    num_partitions: int = 8


@dataclass
class MlflowParams:
    experiment_name: str = "dencast"
    run_name_prefix: str = "run"


@dataclass
class Params:
    """Everything in params.yaml, typed."""

    dataset: DatasetParams
    lsh: LshParams
    clustering: ClusteringParams
    split: SplitParams
    scoring: ScoringParams
    selection: SelectionParams
    evaluation: EvaluationParams
    spark: SparkParams
    mlflow: MlflowParams

    @classmethod
    def load(cls, path: Path | str = PARAMS_PATH) -> Params:
        with open(path, encoding="utf-8") as handle:
            raw: Dict[str, Any] = yaml.safe_load(handle)
        return cls(
            dataset=DatasetParams(**raw["dataset"]),
            lsh=LshParams(**raw.get("lsh", {})),
            clustering=ClusteringParams(**raw.get("clustering", {})),
            split=SplitParams(**raw.get("split", {})),
            scoring=ScoringParams(**raw.get("scoring", {})),
            selection=SelectionParams(**raw.get("selection", {})),
            evaluation=EvaluationParams(**raw.get("evaluation", {})),
            spark=SparkParams(**raw.get("spark", {})),
            mlflow=MlflowParams(**raw.get("mlflow", {})),
        )

    def validate(self) -> None:
        if not self.dataset.feature_cols:
            raise ValueError(
                "dataset has no columns: set dataset.columns (anomaly task) or "
                "dataset.feature_cols and dataset.target_cols (regression)"
            )
        if not self.dataset.columns and not self.dataset.target_cols:
            raise ValueError("dataset.target_cols is empty")
        missing = [c for c in self.scoring.exclude_cols if c not in self.dataset.feature_cols]
        if missing:
            raise ValueError(f"scoring.exclude_cols names columns not in use: {missing}")
        if self.dataset.subsample_minutes is not None and self.dataset.subsample_minutes < 1:
            raise ValueError("dataset.subsample_minutes must be >= 1 when set")
        for col, period in self.dataset.cyclical_cols.items():
            if col not in self.dataset.feature_cols:
                raise ValueError(
                    f"dataset.cyclical_cols names '{col}', which is not in "
                    "feature_cols; a cyclical column is a feature, not an extra one"
                )
            if not period or float(period) <= 0:
                raise ValueError(f"dataset.cyclical_cols['{col}'] must be a positive period")
        if self.clustering.min_pts < 1:
            raise ValueError("clustering.min_pts must be >= 1")
        if not 0.0 <= self.clustering.label_change_rate <= 1.0:
            raise ValueError("clustering.label_change_rate must be in [0, 1]")
        if not -1.0 <= self.lsh.min_sim <= 1.0:
            raise ValueError("lsh.min_sim must be in [-1, 1]")
        if self.lsh.r < 1 or self.lsh.b < 1:
            raise ValueError("lsh.r and lsh.b must be >= 1")
        if self.evaluation.window_size < 1:
            raise ValueError("evaluation.window_size must be >= 1")
        if not 0.0 < self.split.train_fraction < 1.0:
            raise ValueError("split.train_fraction must be strictly between 0 and 1")
        if not 0.0 <= self.split.valid_fraction < 1.0:
            raise ValueError("split.valid_fraction must be in [0, 1)")
        if self.split.test_fraction <= 0.0:
            raise ValueError(
                "split.train_fraction + split.valid_fraction leaves nothing for test "
                f"({self.split.train_fraction} + {self.split.valid_fraction})"
            )
        if self.scoring.scale <= 0.0:
            raise ValueError("scoring.scale must be > 0")
        if self.selection.n_days < 1:
            raise ValueError("selection.n_days must be >= 1")
        if self.evaluation.n_calibration_days < 1:
            raise ValueError("evaluation.n_calibration_days must be >= 1")
        known = {"lsh", "clustering", "scoring"}
        for path in self.selection.grid:
            section, _, field_name = path.partition(".")
            if section not in known or not hasattr(getattr(self, section), field_name):
                raise ValueError(
                    f"selection.grid key '{path}' is not a DENCAST hyperparameter; "
                    f"expected <section>.<field> with section in {sorted(known)}"
                )

    def flat(self) -> Dict[str, Any]:
        """Flatten into `section.key` form, for logging to MLflow."""
        out: Dict[str, Any] = {}
        for section in ("dataset", "lsh", "clustering", "split", "scoring", "evaluation", "spark"):
            # selection.grid is a nested mapping; MLflow takes flat scalars, so
            # the grid is logged as its size and the winner is logged by the
            # stage that picks it.
            block = getattr(self, section)
            for key, value in vars(block).items():
                if isinstance(value, list) and len(value) > 8:
                    # Column lists are long and noisy; the count is what matters.
                    out[f"{section}.{key}_count"] = len(value)
                else:
                    out[f"{section}.{key}"] = value
        return out
