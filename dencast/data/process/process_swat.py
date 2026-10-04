from __future__ import annotations

from pathlib import Path

from dencast.data.process.process import Processor

# SWaT encodes the instrument type in the name, so which columns are categorical is
# stated by the dataset rather than guessed from the data. The convention, as the
# analysis notebook reads it: FIT flow, LIT level, PIT pressure, DPIT differential
# pressure and AIT analyser are sensors reporting a continuous quantity; MV valves,
# P pumps and UV lamps are actuators holding a discrete state.
#
# These are the 20 that survive `preprocess_swat`, which drops the seven instruments
# never recorded before 28/12 -- MV101, MV201, MV303, P201, P202 and P204 of them
# categorical, AIT201 continuous.
SWAT_CATEGORICAL = [
    "MV301", "MV302", "MV304",
    "P101", "P102", "P203", "P205", "P206", "P301", "P302",
    "P401", "P402", "P403", "P404", "P501", "P502", "P601", "P602", "P603",
    "UV401",
]


class SwatProcessor(Processor):
    """`data/interim/swat/split` in, `data/processed/swat` out."""

    def __init__(
        self,
        split_dir: Path = Path("data/interim/swat/split"),
        out_dir: Path = Path("data/processed/swat"),
        name: str = "swat",
        max_levels: int = 5,
        categorical: list[str] | None = SWAT_CATEGORICAL,
    ) -> None:
        super().__init__(split_dir, out_dir, name, max_levels=max_levels,
                         categorical=categorical)


def main() -> None:
    SwatProcessor().run(overwrite=True)


if __name__ == "__main__":
    main()


__all__ = ["SwatProcessor", "SWAT_CATEGORICAL"]
