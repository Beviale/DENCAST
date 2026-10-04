"""Project paths and typed access to params.yaml.

Follows the Cookiecutter Data Science layout: every module resolves paths
through the constants defined here rather than hardcoding them, so the same
code works from the repo root, from a notebook, or from a DVC stage.

The algorithm parameters use the names of the DENCAST paper (Corizzo et al.,
Journal of Big Data 2019), with the original Scala name noted where it differs.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PARAMS_PATH = PROJ_ROOT / "params.yaml"

load_dotenv(PROJ_ROOT / ".env")


