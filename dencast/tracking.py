"""MLflow tracking, wired to DagsHub when credentials are present.

DagsHub exposes an MLflow-compatible tracking server per repository, so nothing
in the training code has to know it is talking to DagsHub: it is a tracking URI
plus basic-auth credentials. That keeps runs reproducible locally -- with no URI
set, MLflow falls back to ./mlruns and everything still works offline.

Configure through .env (see .env.example):

    MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
    MLFLOW_TRACKING_USERNAME=<owner>
    MLFLOW_TRACKING_PASSWORD=<dagshub-token>
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from typing import Any, Dict, Iterator, Optional

from loguru import logger
import mlflow

from dencast.config import PROJ_ROOT
from dencast.utils import Params


def _ensure_utf8_stdout() -> None:
    """Let MLflow print its run links on a Windows console.

    MLflow writes a "View run ... at: <url>" line with a leading emoji when a
    run terminates. The default Windows code page is cp1252, which cannot
    encode it, so the write raises UnicodeEncodeError *after* the run has
    already been logged -- the tracking works, the process dies on the banner.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and getattr(stream, "encoding", "") .lower() not in (
            "utf-8",
            "utf8",
        ):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass


def is_remote_configured() -> bool:
    """True when a tracking URI is set, i.e. runs go to DagsHub rather than disk."""
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def setup(params: Params) -> str:
    """Point MLflow at the right backend and select the experiment.

    Returns the tracking URI in use.
    """
    _ensure_utf8_stdout()

    # MLflow prints a multi-line hint about an assistant skill on every import
    # path that touches tracking. It is noise in a pipeline log and hides the
    # lines that matter.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

    uri = os.getenv("MLFLOW_TRACKING_URI")

    if uri:
        if not (
            os.getenv("MLFLOW_TRACKING_USERNAME")
            and os.getenv("MLFLOW_TRACKING_PASSWORD")
        ):
            logger.warning(
                "MLFLOW_TRACKING_URI is set but the credentials are not. "
                "DagsHub will reject the connection; see .env.example."
            )
        mlflow.set_tracking_uri(uri)
        logger.info("MLflow tracking to {}", uri)
    else:
        # SQLite rather than the ./mlruns file store: recent MLflow refuses the
        # filesystem backend outright ("in maintenance mode"). SQLite is a
        # single local file, needs no server, and supports the full API.
        db = PROJ_ROOT / "mlflow.db"
        uri = f"sqlite:///{db.as_posix()}"
        mlflow.set_tracking_uri(uri)
        logger.info("MLFLOW_TRACKING_URI not set: logging locally to {}", db.name)

    mlflow.set_experiment(params.mlflow.experiment_name)
    return uri


@contextmanager
def start_run(
    params: Params, run_name: str, nested: bool = False, tags: Optional[Dict[str, Any]] = None
) -> Iterator[Any]:
    """Open an MLflow run with the project's parameters already logged."""
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        if tags:
            mlflow.set_tags(tags)
        if not nested:
            # Parameters are immutable per run, so they are logged once on the
            # parent; child runs inherit the context through the run hierarchy.
            mlflow.log_params(params.flat())
        yield run


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log the numeric entries of a dict, skipping anything non-numeric."""
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if numeric:
        mlflow.log_metrics(numeric, step=step)
