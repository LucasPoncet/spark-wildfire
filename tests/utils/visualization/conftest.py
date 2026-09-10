from typing import Never

import matplotlib
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def forbid_savefig_inside_plotters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any plotter that writes to disk.

    `Responsability_file.md` gives every function in `src/utils/visualization/`
    the same shape: take data, return a Figure. Saving belongs to `scripts/`.
    Monkeypatching `savefig` to raise turns that rule into something the suite
    enforces rather than something the reviewer has to notice.
    """

    def refuse_to_save(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("a plotter must never call savefig")

    monkeypatch.setattr(Figure, "savefig", refuse_to_save)
