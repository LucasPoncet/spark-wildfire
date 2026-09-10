"""What the receivers heard, and what the estimator made of it.

Takes data and returns a `matplotlib.figure.Figure`. Never calls `savefig`.
Every function that needs both forward and inverse quantities takes both as
explicit arguments, so nothing here imports `inverse/` or `fire/` and reaches
for state on its own.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from src.utils.array_types import Float64Array

ELLIPSE_AXIS_SCALE: float = 2.0
CHI_SQUARE_REFERENCE: float = 1.0
CHI_SQUARE_CONSISTENT_LOWER: float = 0.5
CHI_SQUARE_CONSISTENT_UPPER: float = 2.0
DEGENERATE_MARKER_HEIGHT: float = 0.2
ZOOM_MARGIN_FACTOR: float = 1.6
MINIMUM_ZOOM_HALF_WIDTH_M: float = 0.05
ESTIMATE_COLOR: str = "#2f5d8a"
TRUTH_COLOR: str = "#e08a1e"
RECEIVER_COLOR: str = "#444444"


def plot_receiver_level_traces(
    observation_times_s: Float64Array,
    receiver_levels_db: Float64Array,
    receiver_positions_xy_m: Float64Array,
) -> Figure:
    """Received level at each receiver over the observations of one run.

    Args:
        observation_times_s: Float64 array of shape (n_observations,).
        receiver_levels_db: Float64 array of shape (n_observations, n_receivers).
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).

    Returns:
        A matplotlib Figure. The caller saves or displays it.

    Raises:
        ValueError: If the level array does not match the times and receivers.
    """
    if receiver_levels_db.shape != (
        observation_times_s.size,
        receiver_positions_xy_m.shape[0],
    ):
        raise ValueError(
            "receiver levels must have shape (n_observations, n_receivers), got "
            f"{receiver_levels_db.shape}"
        )

    figure = Figure(figsize=(3.5, 2.8))
    axes = figure.add_subplot(111)
    for receiver_index in range(receiver_positions_xy_m.shape[0]):
        position_xy_m = receiver_positions_xy_m[receiver_index]
        axes.plot(
            observation_times_s,
            receiver_levels_db[:, receiver_index],
            label=f"({position_xy_m[0]:.0f}, {position_xy_m[1]:.0f}) m",
        )
    axes.set_xlabel("simulation time (s)")
    axes.set_ylabel("received level (dB)")
    axes.set_title("Receiver levels over the run")
    axes.legend(fontsize="xx-small", ncol=2)
    axes.grid(True, linewidth=0.3, alpha=0.5)
    figure.tight_layout()
    return figure


def plot_band_analysis_chain(
    band_center_frequencies_hz: Float64Array,
    receiver_1_levels_db: Float64Array,
    receiver_2_levels_db: Float64Array,
    absorption_coefficients_db_per_m: Float64Array,
    path_difference_m: float,
) -> Figure:
    """The estimator's per-band chain on one clip, made visible.

    Three panels: the two receivers' band levels, their difference, and the
    geometric term left after the absorption already explained by the path
    difference is subtracted.

    Args:
        band_center_frequencies_hz: One frequency per band, in hertz.
        receiver_1_levels_db: Band levels at receiver 1, in decibels.
        receiver_2_levels_db: Band levels at receiver 2, in decibels.
        absorption_coefficients_db_per_m: One coefficient per band, in dB/m.
        path_difference_m: Range difference from the delay, in metres.

    Returns:
        A matplotlib Figure with three axes.
    """
    level_differences_db = receiver_1_levels_db - receiver_2_levels_db
    absorption_term_db = absorption_coefficients_db_per_m * path_difference_m
    geometric_terms_db = level_differences_db - absorption_term_db

    figure = Figure(figsize=(7.0, 2.6))
    level_axes = figure.add_subplot(131)
    level_axes.semilogx(
        band_center_frequencies_hz, receiver_1_levels_db, marker="o", label="receiver 1"
    )
    level_axes.semilogx(
        band_center_frequencies_hz, receiver_2_levels_db, marker="s", label="receiver 2"
    )
    level_axes.set_xlabel("band centre (Hz)")
    level_axes.set_ylabel("band level (dB)")
    level_axes.set_title("Band levels")
    level_axes.legend(fontsize="xx-small")
    level_axes.grid(True, which="both", linewidth=0.3, alpha=0.5)

    difference_axes = figure.add_subplot(132)
    difference_axes.semilogx(
        band_center_frequencies_hz, level_differences_db, marker="o", color="#2f5d8a"
    )
    difference_axes.semilogx(
        band_center_frequencies_hz,
        absorption_term_db,
        marker="^",
        linestyle="--",
        color="#d94a3d",
        label=r"$\alpha_b D$",
    )
    difference_axes.set_xlabel("band centre (Hz)")
    difference_axes.set_ylabel(r"$\Delta L_b$ (dB)")
    difference_axes.set_title("Level difference")
    difference_axes.legend(fontsize="xx-small")
    difference_axes.grid(True, which="both", linewidth=0.3, alpha=0.5)

    geometric_axes = figure.add_subplot(133)
    geometric_axes.semilogx(
        band_center_frequencies_hz, geometric_terms_db, marker="o", color="#3f7d4f"
    )
    geometric_axes.axhline(
        float(np.mean(geometric_terms_db)),
        linestyle="--",
        linewidth=0.8,
        color="#666666",
    )
    geometric_axes.set_xlabel("band centre (Hz)")
    geometric_axes.set_ylabel(r"$G_b$ (dB)")
    geometric_axes.set_title("Geometric term")
    geometric_axes.grid(True, which="both", linewidth=0.3, alpha=0.5)

    figure.tight_layout()
    return figure


def compute_zoom_half_width_m(
    estimated_positions_xy_m: Float64Array,
    true_position_xy_m: Float64Array,
    minimum_half_width_m: float,
) -> float:
    """Half-width of an inset that comfortably holds every estimate.

    Args:
        estimated_positions_xy_m: Float64 array of shape (n_clips, 2).
        true_position_xy_m: Float64 array of shape (2,).
        minimum_half_width_m: Floor, so a perfect estimate still gets a window.

    Returns:
        Half-width in metres, square so the inset preserves aspect.
    """
    if estimated_positions_xy_m.shape[0] == 0:
        return minimum_half_width_m
    offsets_m = np.abs(estimated_positions_xy_m - true_position_xy_m)
    return float(max(ZOOM_MARGIN_FACTOR * offsets_m.max(), minimum_half_width_m))


def draw_estimate_cloud(
    axes: Axes,
    estimated_positions_xy_m: Float64Array,
    ellipse_semi_major_m: Float64Array,
    ellipse_semi_minor_m: Float64Array,
    true_position_xy_m: Float64Array,
    marker_size: float,
) -> None:
    """Draw the estimates, their ellipses and the truth on one axes.

    Truth is a hollow ring with a dark edge drawn above everything else, so it
    stays legible when the estimates land on top of it — which is the normal
    case here, where errors are centimetres against a domain of a hundred
    metres.

    Args:
        axes: The axes to draw on.
        estimated_positions_xy_m: Float64 array of shape (n_clips, 2).
        ellipse_semi_major_m: One-sigma major semi-axis per clip, in metres.
        ellipse_semi_minor_m: One-sigma minor semi-axis per clip, in metres.
        true_position_xy_m: Float64 array of shape (2,).
        marker_size: Area of the estimate markers, in points squared.
    """
    for clip_index in range(estimated_positions_xy_m.shape[0]):
        axes.add_patch(
            Ellipse(
                xy=(
                    float(estimated_positions_xy_m[clip_index, 0]),
                    float(estimated_positions_xy_m[clip_index, 1]),
                ),
                width=ELLIPSE_AXIS_SCALE * float(ellipse_semi_major_m[clip_index]),
                height=ELLIPSE_AXIS_SCALE * float(ellipse_semi_minor_m[clip_index]),
                facecolor=ESTIMATE_COLOR,
                edgecolor="none",
                alpha=0.15,
            )
        )
    axes.scatter(
        estimated_positions_xy_m[:, 0],
        estimated_positions_xy_m[:, 1],
        s=marker_size,
        facecolor=ESTIMATE_COLOR,
        edgecolor="white",
        linewidth=0.4,
        label="estimates",
        zorder=3,
    )
    axes.scatter(
        [true_position_xy_m[0]],
        [true_position_xy_m[1]],
        marker="o",
        s=4.0 * marker_size,
        facecolor="none",
        edgecolor=TRUTH_COLOR,
        linewidth=1.6,
        label="truth",
        zorder=5,
    )
    axes.plot(
        [true_position_xy_m[0]],
        [true_position_xy_m[1]],
        marker="+",
        markersize=6,
        markeredgewidth=1.2,
        color=TRUTH_COLOR,
        linestyle="none",
        zorder=6,
    )


def plot_estimates_against_truth(
    estimated_positions_xy_m: Float64Array,
    ellipse_semi_major_m: Float64Array,
    ellipse_semi_minor_m: Float64Array,
    true_position_xy_m: Float64Array,
    receiver_positions_xy_m: Float64Array,
    title: str,
) -> Figure:
    """Estimates in the plane against ground truth, with error ellipses.

    Two panels, because one cannot show both facts at once. The left panel is
    the scene at domain scale, where the estimates and the truth sit on top of
    each other — that overlap *is* the result. The right panel zooms to the
    spread of the estimates, where the residual scatter becomes readable and
    the scale bar says how small it is.

    Args:
        estimated_positions_xy_m: Float64 array of shape (n_clips, 2).
        ellipse_semi_major_m: One-sigma major semi-axis per clip, in metres.
        ellipse_semi_minor_m: One-sigma minor semi-axis per clip, in metres.
        true_position_xy_m: Float64 array of shape (2,).
        receiver_positions_xy_m: Float64 array of shape (n_receivers, 2).
        title: Figure title, naming the scenario.

    Returns:
        A matplotlib Figure with two axes.
    """
    figure = Figure(figsize=(7.0, 3.2))
    scene_axes = figure.add_subplot(121)
    zoom_axes = figure.add_subplot(122)

    draw_estimate_cloud(
        scene_axes,
        estimated_positions_xy_m,
        ellipse_semi_major_m,
        ellipse_semi_minor_m,
        true_position_xy_m,
        marker_size=14.0,
    )
    scene_axes.scatter(
        receiver_positions_xy_m[:, 0],
        receiver_positions_xy_m[:, 1],
        marker="v",
        s=30,
        color=RECEIVER_COLOR,
        label="receivers",
        zorder=3,
    )
    scene_axes.set_xlabel("x (m)")
    scene_axes.set_ylabel("y (m)")
    scene_axes.set_title(f"{title} — scene")
    scene_axes.legend(fontsize="x-small")
    scene_axes.grid(True, linewidth=0.3, alpha=0.5)

    half_width_m = compute_zoom_half_width_m(
        estimated_positions_xy_m, true_position_xy_m, MINIMUM_ZOOM_HALF_WIDTH_M
    )
    draw_estimate_cloud(
        zoom_axes,
        estimated_positions_xy_m,
        ellipse_semi_major_m,
        ellipse_semi_minor_m,
        true_position_xy_m,
        marker_size=36.0,
    )
    zoom_axes.set_xlim(
        float(true_position_xy_m[0]) - half_width_m,
        float(true_position_xy_m[0]) + half_width_m,
    )
    zoom_axes.set_ylim(
        float(true_position_xy_m[1]) - half_width_m,
        float(true_position_xy_m[1]) + half_width_m,
    )
    zoom_axes.set_aspect("equal")
    zoom_axes.set_xlabel("x (m)")
    zoom_axes.set_ylabel("y (m)")
    zoom_axes.set_title(f"zoom, ±{half_width_m:.2f} m about truth")
    zoom_axes.grid(True, linewidth=0.3, alpha=0.5)

    errors_m = np.linalg.norm(estimated_positions_xy_m - true_position_xy_m, axis=1)
    if errors_m.size > 0:
        zoom_axes.annotate(
            f"median error {np.median(errors_m) * 100.0:.1f} cm",
            xy=(0.03, 0.03),
            xycoords="axes fraction",
            fontsize="x-small",
            color="#444444",
        )

    figure.tight_layout()
    return figure


def plot_error_against_signal_to_noise_ratio(
    signal_to_noise_ratios_db: Float64Array,
    errors_m: Float64Array,
    scenario_labels: list[str],
) -> Figure:
    """Position error against signal-to-noise ratio, one line per scenario.

    Args:
        signal_to_noise_ratios_db: One ratio per column, in decibels.
        errors_m: Float64 array of shape (n_scenarios, n_ratios), in metres.
        scenario_labels: One label per scenario.

    Returns:
        A matplotlib Figure. The caller saves or displays it.

    Raises:
        ValueError: If the error array does not match the labels and ratios.
    """
    if errors_m.shape != (len(scenario_labels), signal_to_noise_ratios_db.size):
        raise ValueError(
            f"errors must have shape (n_scenarios, n_ratios), got {errors_m.shape}"
        )

    figure = Figure(figsize=(3.5, 2.8))
    axes = figure.add_subplot(111)
    for scenario_index, label in enumerate(scenario_labels):
        axes.semilogy(
            signal_to_noise_ratios_db,
            errors_m[scenario_index],
            marker="o",
            label=label,
        )
    axes.set_xlabel("signal-to-noise ratio (dB)")
    axes.set_ylabel("position error (m)")
    axes.set_title("Error against noise")
    axes.legend(fontsize="xx-small")
    axes.grid(True, which="both", linewidth=0.3, alpha=0.5)
    figure.tight_layout()
    return figure


def plot_error_against_bisector_distance(
    bisector_distances_m: Float64Array,
    errors_m: Float64Array,
    near_singular_band_half_width_m: float,
) -> Figure:
    """Position error against distance from the perpendicular bisector.

    The shaded band is the degenerate region. This is the identifiability
    figure: it reports where the estimator cannot work even in principle, and
    it is informative whether or not the errors outside the band are small.

    Args:
        bisector_distances_m: Signed or unsigned distance per estimate, in metres.
        errors_m: Position error per estimate, in metres.
        near_singular_band_half_width_m: Half width of the degenerate band.

    Returns:
        A matplotlib Figure. The caller saves or displays it.
    """
    figure = Figure(figsize=(3.5, 2.8))
    axes = figure.add_subplot(111)
    axes.axvspan(
        0.0,
        near_singular_band_half_width_m,
        color="#d94a3d",
        alpha=0.15,
        label="near-singular",
    )
    axes.semilogy(
        bisector_distances_m,
        np.maximum(errors_m, np.finfo(np.float64).tiny),
        linestyle="none",
        marker="o",
        markersize=4,
        color="#2f5d8a",
    )
    axes.set_xlabel("distance from perpendicular bisector (m)")
    axes.set_ylabel("position error (m)")
    axes.set_title("Error against degeneracy")
    axes.legend(fontsize="x-small")
    axes.grid(True, which="both", linewidth=0.3, alpha=0.5)
    figure.tight_layout()
    return figure


def plot_reduced_chi_square_by_scenario(
    scenario_labels: list[str],
    reduced_chi_squares: list[Float64Array],
    is_degenerate: list[bool] | None = None,
) -> Figure:
    """Reduced chi-square per scenario, against the value a calibrated fit gives.

    Reduced chi-square is the estimator's own consistency check. Each octave
    band gives an independent estimate of the same geometric level difference,
    with a variance the estimator predicts for it. Reduced chi-square is the
    spread the bands actually show divided by the spread they were predicted to
    show, per degree of freedom.

    One means the bands disagree by exactly as much as the stated uncertainties
    allow, so an error ellipse can be believed. Far above one means the bands
    disagree far more than predicted: either the uncertainties are understated
    or the model is missing a term that varies band to band. Far below one means
    the uncertainties are overstated. It says nothing about whether the position
    is right — only about whether the error bar on it is.

    A scenario flagged degenerate is drawn as a marker on the axis floor rather
    than as a box. Its chi-square is not large or small, it is meaningless: on
    the perpendicular bisector both observables vanish, so there is no fit for
    the bands to agree or disagree about. Dropping it silently would leave a
    blank column that reads as missing data.

    Args:
        scenario_labels: One label per scenario.
        reduced_chi_squares: One array of per-clip values per scenario.
        is_degenerate: One flag per scenario, True where every clip was flagged
            near-singular. None treats every scenario as well posed.

    Returns:
        A matplotlib Figure. The caller saves or displays it.

    Raises:
        ValueError: If the labels, value arrays and flags disagree in length.
    """
    if len(scenario_labels) != len(reduced_chi_squares):
        raise ValueError("every scenario needs exactly one label")
    degenerate_flags = (
        [False] * len(scenario_labels) if is_degenerate is None else is_degenerate
    )
    if len(degenerate_flags) != len(scenario_labels):
        raise ValueError("every scenario needs exactly one label")

    figure = Figure(figsize=(4.2, 3.0))
    axes = figure.add_subplot(111)
    axes.axhspan(
        CHI_SQUARE_CONSISTENT_LOWER,
        CHI_SQUARE_CONSISTENT_UPPER,
        color="#3f7d4f",
        alpha=0.12,
        label="consistent",
    )
    axes.axhline(
        CHI_SQUARE_REFERENCE,
        linestyle="--",
        linewidth=0.9,
        color="#3f7d4f",
    )
    well_posed_positions = [
        position
        for position, degenerate in enumerate(degenerate_flags, start=1)
        if not degenerate
    ]
    if well_posed_positions:
        axes.boxplot(
            [
                np.asarray(values, dtype=np.float64)
                for values, degenerate in zip(
                    reduced_chi_squares, degenerate_flags, strict=True
                )
                if not degenerate
            ],
            positions=well_posed_positions,
        )
    degenerate_positions = [
        position
        for position, degenerate in enumerate(degenerate_flags, start=1)
        if degenerate
    ]
    if degenerate_positions:
        axes.scatter(
            degenerate_positions,
            [DEGENERATE_MARKER_HEIGHT] * len(degenerate_positions),
            marker="x",
            s=48,
            color="#d94a3d",
            label="degenerate, no fit",
            zorder=4,
        )
    axes.set_xticks(range(1, len(scenario_labels) + 1))
    axes.set_xticklabels(scenario_labels)
    axes.set_xlim(0.5, len(scenario_labels) + 0.5)
    axes.set_yscale("log")
    axes.set_ylabel(r"reduced $\chi^2$")
    axes.set_title("Uncertainty calibration")
    axes.tick_params(axis="x", labelrotation=45, labelsize="x-small")
    axes.legend(fontsize="x-small", loc="lower right")
    axes.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.5)
    axes.annotate(
        "1 = bands disagree exactly as much as the stated uncertainties allow\n"
        "above 1 = uncertainties understated, or a missing band-dependent term",
        xy=(0.0, -0.34),
        xycoords="axes fraction",
        fontsize="xx-small",
        color="#444444",
        verticalalignment="top",
    )
    figure.tight_layout()
    return figure


def plot_front_position_over_observations(
    observation_times_s: Float64Array,
    true_front_radius_m: Float64Array,
    estimated_front_radius_m: Float64Array | None,
) -> Figure:
    """Front travel over the run, with a linear rate-of-spread fit and residual.

    Args:
        observation_times_s: Float64 array of shape (n_observations,).
        true_front_radius_m: Ground-truth mean front travel, in metres.
        estimated_front_radius_m: Estimated travel, or None when the estimator
            has not been run on this scene yet.

    Returns:
        A matplotlib Figure with two axes.
    """
    slope_m_per_s, intercept_m = np.polyfit(observation_times_s, true_front_radius_m, 1)
    fitted_radius_m = slope_m_per_s * observation_times_s + intercept_m

    figure = Figure(figsize=(7.0, 2.8))
    travel_axes = figure.add_subplot(121)
    travel_axes.plot(
        observation_times_s, true_front_radius_m, marker="o", label="ground truth"
    )
    if estimated_front_radius_m is not None:
        travel_axes.plot(
            observation_times_s,
            estimated_front_radius_m,
            marker="s",
            linestyle="none",
            label="estimated",
        )
    travel_axes.plot(
        observation_times_s,
        fitted_radius_m,
        linestyle="--",
        color="#666666",
        label=f"fit {slope_m_per_s * 1000.0:.2f} mm/s",
    )
    travel_axes.set_xlabel("simulation time (s)")
    travel_axes.set_ylabel("front travel (m)")
    travel_axes.set_title("Front position")
    travel_axes.legend(fontsize="xx-small")
    travel_axes.grid(True, linewidth=0.3, alpha=0.5)

    residual_axes = figure.add_subplot(122)
    residual_axes.axhline(0.0, linewidth=0.8, color="#666666")
    residual_axes.plot(
        observation_times_s,
        true_front_radius_m - fitted_radius_m,
        marker="o",
        color="#2f5d8a",
    )
    residual_axes.set_xlabel("simulation time (s)")
    residual_axes.set_ylabel("residual (m)")
    residual_axes.set_title("Rate-of-spread fit residual")
    residual_axes.grid(True, linewidth=0.3, alpha=0.5)

    figure.tight_layout()
    return figure


def plot_pair_correlation_curves(
    lag_axes_s: list[Float64Array],
    correlation_values: list[Float64Array],
    pair_labels: list[str],
    true_delays_s: Float64Array,
    title: str,
) -> Figure:
    """Draws one generalised cross-correlation per receiver pair.

    The true delay is marked on each, so a peak that sits somewhere else is
    visible as such rather than being averaged into a position error.

    Args:
        lag_axes_s: Lag axis per pair, each shape `(n_lags,)`.
        correlation_values: Curve per pair, matching shapes.
        pair_labels: One label per pair.
        true_delays_s: True delay per pair, shape `(n_pairs,)`. Pass an empty
            array when no ground truth is available.
        title: Figure title.

    Returns:
        The figure. Saving it is the caller's job.

    Raises:
        ValueError: If the axis, value and label counts disagree.
    """
    if not len(lag_axes_s) == len(correlation_values) == len(pair_labels):
        raise ValueError("one lag axis and one label are required per curve")
    pair_count = len(pair_labels)
    figure = Figure(figsize=(7.5, 1.6 * pair_count + 1.0))
    pair_axes = [
        figure.add_subplot(pair_count, 1, index + 1) for index in range(pair_count)
    ]
    delays_s = np.asarray(true_delays_s, dtype=np.float64)
    for index, (lags_s, values, label) in enumerate(
        zip(lag_axes_s, correlation_values, pair_labels, strict=True)
    ):
        axes = pair_axes[index]
        axes.plot(1e3 * np.asarray(lags_s), np.asarray(values), linewidth=0.9)
        if delays_s.size > index:
            axes.axvline(
                1e3 * float(delays_s[index]),
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
            )
        axes.set_ylabel(label, fontsize=8)
        axes.grid(visible=True, alpha=0.3)
    pair_axes[-1].set_xlabel("lag (ms)")
    pair_axes[0].set_title(title)
    figure.tight_layout()
    return figure


def plot_band_level_difference_regression(
    band_center_frequencies_hz: Float64Array,
    absorption_coefficients_db_per_m: Float64Array,
    band_level_differences_db: Float64Array,
    path_difference_m: float,
    fused_geometric_level_difference_db: float,
    title: str,
) -> Figure:
    """Draws the per-band level difference against its absorption coefficient.

    The model says the measured difference is a common geometric offset plus a
    term proportional to the coefficient, so the points should fall on a line
    whose intercept is the geometric term and whose slope is the path
    difference. A curved scatter is the model failing, not noise.

    Args:
        band_center_frequencies_hz: Band centres, shape `(n_bands,)`.
        absorption_coefficients_db_per_m: One coefficient per band.
        band_level_differences_db: Measured difference per band.
        path_difference_m: Range difference the delay implies, in metres.
        fused_geometric_level_difference_db: Fused geometric term, in decibels.
        title: Figure title.

    Returns:
        The figure. Saving it is the caller's job.
    """
    coefficients = np.asarray(absorption_coefficients_db_per_m, dtype=np.float64)
    differences_db = np.asarray(band_level_differences_db, dtype=np.float64)
    centres_hz = np.asarray(band_center_frequencies_hz, dtype=np.float64)

    figure = Figure(figsize=(7.0, 4.5))
    axes = figure.add_subplot(111)
    scatter = axes.scatter(
        coefficients, differences_db, c=centres_hz, cmap="viridis", s=60
    )
    figure.colorbar(scatter, ax=axes, label="band centre (Hz)")
    predicted_db = (
        fused_geometric_level_difference_db + coefficients * path_difference_m
    )
    order = np.argsort(coefficients)
    axes.plot(
        coefficients[order],
        predicted_db[order],
        color="tab:red",
        linewidth=1.2,
        label="fitted model",
    )
    axes.set_xlabel("absorption coefficient (dB/m)")
    axes.set_ylabel("level difference (dB)")
    axes.set_title(title)
    axes.grid(visible=True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure
