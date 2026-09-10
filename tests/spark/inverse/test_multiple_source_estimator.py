import numpy as np
import pytest

from src.config.multi_source_localization_configuration import (
    MultiSourceLocalizationConfiguration,
)
from src.spark.atmosphere.atmospheric_conditions import AtmosphericConditions
from src.spark.inverse.multiple_source_estimator import (
    STOPPED_ON_PROMINENCE,
    STOPPED_ON_SEPARATION,
    MultipleSourceEstimator,
    is_far_enough_from_accepted,
    localize_multiple_sources,
)
from src.utils.array_types import Float64Array
from src.utils.metrics.localization_metrics import match_estimated_to_true_sources
from tests.spark.inverse.conftest import (
    SCENE_DOMAIN_EXTENT_M,
    SCENE_SOURCE_HEIGHT_M,
    render_scene_signals,
)

RECOVERY_TOLERANCE_M: float = 2.0


def localize(
    source_positions_xy_m: Float64Array,
    source_amplitude_scales: Float64Array,
    receivers_xyz_m: Float64Array,
    conditions: AtmosphericConditions,
    multi_source: MultiSourceLocalizationConfiguration,
    sample_rate_hz: int,
    seed: int = 0,
) -> tuple[Float64Array, str, int]:
    localization = localize_multiple_sources(
        render_scene_signals(
            source_positions_xy_m,
            source_amplitude_scales,
            receivers_xyz_m,
            conditions,
            sample_rate_hz,
            seed,
        ),
        receivers_xyz_m,
        sample_rate_hz,
        SCENE_DOMAIN_EXTENT_M,
        SCENE_DOMAIN_EXTENT_M,
        conditions,
        multi_source,
    )
    estimated_positions_xy_m = (
        np.stack([source.position.position_xy_m for source in localization.sources])
        if localization.sources
        else np.empty((0, 2))
    )
    return (
        estimated_positions_xy_m,
        localization.stop_reason,
        localization.estimated_source_count,
    )


def test_one_source_is_recovered_and_the_loop_then_stops(
    scene_receivers_xyz_m: Float64Array,
    atmosphere: AtmosphericConditions,
    multi_source: MultiSourceLocalizationConfiguration,
    sample_rate_hz: int,
) -> None:
    true_positions_xy_m = np.array([[12.0, 27.0]])
    estimates_xy_m, stop_reason, source_count = localize(
        true_positions_xy_m,
        np.array([1.0]),
        scene_receivers_xyz_m,
        atmosphere,
        multi_source,
        sample_rate_hz,
    )
    assert source_count == 1
    assert stop_reason in (STOPPED_ON_SEPARATION, STOPPED_ON_PROMINENCE)
    assert float(np.linalg.norm(estimates_xy_m[0] - true_positions_xy_m[0])) < (
        RECOVERY_TOLERANCE_M
    )


def test_two_equally_loud_sources_are_both_recovered(
    scene_receivers_xyz_m: Float64Array,
    atmosphere: AtmosphericConditions,
    multi_source: MultiSourceLocalizationConfiguration,
    sample_rate_hz: int,
) -> None:
    true_positions_xy_m = np.array([[12.0, 27.0], [29.0, 13.0]])
    estimates_xy_m, _, source_count = localize(
        true_positions_xy_m,
        np.array([1.0, 1.0]),
        scene_receivers_xyz_m,
        atmosphere,
        multi_source,
        sample_rate_hz,
    )
    assert source_count == 2
    assignment = match_estimated_to_true_sources(true_positions_xy_m, estimates_xy_m)
    assert assignment.detection_count == 2
    assert float(np.max(assignment.matched_errors_m)) < RECOVERY_TOLERANCE_M


def test_the_source_count_is_an_output_not_an_input(
    scene_receivers_xyz_m: Float64Array,
    atmosphere: AtmosphericConditions,
    multi_source: MultiSourceLocalizationConfiguration,
    sample_rate_hz: int,
) -> None:
    """The same settings return one source for one and two for two."""
    _, _, one = localize(
        np.array([[12.0, 27.0]]),
        np.array([1.0]),
        scene_receivers_xyz_m,
        atmosphere,
        multi_source,
        sample_rate_hz,
        seed=1,
    )
    _, _, two = localize(
        np.array([[12.0, 27.0], [29.0, 13.0]]),
        np.array([1.0, 1.0]),
        scene_receivers_xyz_m,
        atmosphere,
        multi_source,
        sample_rate_hz,
        seed=1,
    )
    assert one == 1
    assert two == 2


def test_the_estimator_class_and_the_driver_agree(
    scene_receivers_xyz_m: Float64Array,
    atmosphere: AtmosphericConditions,
    multi_source: MultiSourceLocalizationConfiguration,
    sample_rate_hz: int,
) -> None:
    receiver_signals = render_scene_signals(
        np.array([[12.0, 27.0]]),
        np.array([1.0]),
        scene_receivers_xyz_m,
        atmosphere,
        sample_rate_hz,
        seed=2,
    )
    estimator = MultipleSourceEstimator(
        receiver_positions_xyz_m=scene_receivers_xyz_m,
        domain_extent_x_m=SCENE_DOMAIN_EXTENT_M,
        domain_extent_y_m=SCENE_DOMAIN_EXTENT_M,
        conditions=atmosphere,
        configuration=multi_source,
    )
    from_class = estimator.estimate(receiver_signals, sample_rate_hz)
    from_driver = localize_multiple_sources(
        receiver_signals,
        scene_receivers_xyz_m,
        sample_rate_hz,
        SCENE_DOMAIN_EXTENT_M,
        SCENE_DOMAIN_EXTENT_M,
        atmosphere,
        multi_source,
    )
    assert from_class.estimated_source_count == from_driver.estimated_source_count
    assert np.allclose(
        from_class.sources[0].position.position_xy_m,
        from_driver.sources[0].position.position_xy_m,
    )


def test_a_candidate_beside_an_accepted_source_is_not_a_new_source() -> None:
    from src.spark.inverse.multilateration import PositionEstimate
    from src.spark.inverse.multiple_source_estimator import LocatedSource

    accepted = [
        LocatedSource(
            position=PositionEstimate(
                position_xy_m=np.array([20.0, 20.0]),
                position_covariance_m2=np.eye(2),
                residual_delays_s=np.zeros(1),
                reduced_chi_square=1.0,
                degrees_of_freedom=1,
                iteration_count=1,
                has_converged=True,
                is_ambiguous=False,
            ),
            map_position_xyz_m=np.array([20.0, 20.0, SCENE_SOURCE_HEIGHT_M]),
            map_peak_value=1.0,
            map_peak_to_median_ratio=10.0,
            residual_energy_ratio=0.5,
            pair_delays_s=np.zeros(1),
            pair_delay_variances_s2=np.ones(1),
        )
    ]
    assert not is_far_enough_from_accepted(
        np.array([21.0, 20.0, SCENE_SOURCE_HEIGHT_M]), accepted, 3.0
    )
    assert is_far_enough_from_accepted(
        np.array([30.0, 20.0, SCENE_SOURCE_HEIGHT_M]), accepted, 3.0
    )


def test_the_first_round_map_is_kept_for_display(
    scene_receivers_xyz_m: Float64Array,
    atmosphere: AtmosphericConditions,
    multi_source: MultiSourceLocalizationConfiguration,
    sample_rate_hz: int,
) -> None:
    localization = localize_multiple_sources(
        render_scene_signals(
            np.array([[12.0, 27.0]]),
            np.array([1.0]),
            scene_receivers_xyz_m,
            atmosphere,
            sample_rate_hz,
            seed=4,
        ),
        scene_receivers_xyz_m,
        sample_rate_hz,
        SCENE_DOMAIN_EXTENT_M,
        SCENE_DOMAIN_EXTENT_M,
        atmosphere,
        multi_source,
    )
    assert localization.first_round_map.size > 0
    assert (
        localization.first_round_positions_xyz_m.shape[0]
        == localization.first_round_map.size
    )
    assert localization.speed_of_sound_m_per_s == pytest.approx(340.0, abs=5.0)
