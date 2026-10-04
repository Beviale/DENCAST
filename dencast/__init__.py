"""DENCAST: distributed density-based clustering for multi-target regression.

A PySpark port of

    Corizzo R., Pio G., Ceci M., Malerba D.
    "DENCAST: distributed density-based clustering for multi-target regression"
    Journal of Big Data 6:43 (2019)
    https://doi.org/10.1186/s40537-019-0207-2

Project layout follows Cookiecutter Data Science; the pipeline is orchestrated
by DVC (see dvc.yaml) and every run is tracked with MLflow.
"""

from dencast.utils import Params

__all__ = ["Params"]
__version__ = "0.2.0"
