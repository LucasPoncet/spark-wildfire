import numpy as np
import pytest
from matplotlib.figure import Figure

from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.utils.visualization.channel_plotter import (
    compute_total_channel_gain_db,
    plot_atmospheric_absorption,
    plot_channel_gain_against_range,
    plot_channel_panels,
)

FREQUENCIES_HZ = np.geomspace(50.0, 16000.0, 60)
BAND_CENTRES_HZ = np.array([125.0, 500.0, 2000.0, 8000.0])
RANGES_M = np.linspace(1.0, 120.0, 40)
REFERENCE_DISTANCE_M = 1.0
CONDITIONS = AtmosphericConditions(15.0, 70.0, 101.325)


def test_gain_falls_with_range_in_every_band() -> None:
    gains_db = compute_total_channel_gain_db(
        BAND_CENTRES_HZ, RANGES_M, CONDITIONS, REFERENCE_DISTANCE_M
    )
    assert gains_db.shape == (BAND_CENTRES_HZ.size, RANGES_M.size)
    assert bool(np.all(np.diff(gains_db, axis=1) <= 1e-9))


def test_high_bands_lose_more_than_low_bands_at_range() -> None:
    gains_db = compute_total_channel_gain_db(
        BAND_CENTRES_HZ, RANGES_M, CONDITIONS, REFERENCE_DISTANCE_M
    )
    assert gains_db[-1, -1] < gains_db[0, -1]


def test_the_spread_between_bands_grows_with_range() -> None:
    gains_db = compute_total_channel_gain_db(
        BAND_CENTRES_HZ, RANGES_M, CONDITIONS, REFERENCE_DISTANCE_M
    )
    near_spread_db = gains_db[0, 0] - gains_db[-1, 0]
    far_spread_db = gains_db[0, -1] - gains_db[-1, -1]
    assert far_spread_db > near_spread_db


def test_the_absorption_figure_has_a_curve_per_air_state() -> None:
    figure = plot_atmospheric_absorption(
        FREQUENCIES_HZ,
        [CONDITIONS, AtmosphericConditions(30.0, 20.0, 101.325)],
        ["15 °C, 70 %", "30 °C, 20 %"],
    )
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert len(figure.axes[0].get_lines()) == 2


def test_the_absorption_figure_rejects_missing_labels() -> None:
    with pytest.raises(ValueError, match="exactly one label"):
        plot_atmospheric_absorption(FREQUENCIES_HZ, [CONDITIONS], [])


def test_the_gain_figure_has_a_curve_per_band() -> None:
    figure = plot_channel_gain_against_range(
        BAND_CENTRES_HZ, RANGES_M, CONDITIONS, REFERENCE_DISTANCE_M
    )
    assert len(figure.axes) == 1
    assert len(figure.axes[0].get_lines()) == BAND_CENTRES_HZ.size


def test_the_panel_figure_has_two_axes() -> None:
    figure = plot_channel_panels(
        FREQUENCIES_HZ,
        BAND_CENTRES_HZ,
        RANGES_M,
        CONDITIONS,
        REFERENCE_DISTANCE_M,
    )
    assert len(figure.axes) == 2


def test_the_axes_carry_unit_labels() -> None:
    figure = plot_channel_gain_against_range(
        BAND_CENTRES_HZ, RANGES_M, CONDITIONS, REFERENCE_DISTANCE_M
    )
    axes = figure.axes[0]
    assert "(m)" in axes.get_xlabel()
    assert "dB" in axes.get_ylabel()
