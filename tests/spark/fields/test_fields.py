import numpy as np
import pytest

from src.spark.fields.constant_wind_field import ConstantWindField
from src.spark.fields.patchy_density_field import PatchyDensityField
from src.spark.fields.random_tree_placement_field import RandomTreePlacementField
from src.spark.fields.scalar_field_protocol import ScalarFieldProtocol
from src.spark.fields.uniform_scalar_field import UniformScalarField
from src.spark.fields.uniform_vector_field import UniformVectorField
from src.spark.fields.vector_field_protocol import VectorFieldProtocol

POSITIONS_XYZ = np.array(
    [[0.0, 0.0, 0.0], [10.0, 5.0, 0.0], [-3.0, 7.0, 2.0]], dtype=np.float64
)


def test_uniform_scalar_field_returns_the_constant_at_every_position() -> None:
    field = UniformScalarField(0.5)
    values = field.sample(POSITIONS_XYZ)
    assert values.shape == (3,)
    np.testing.assert_array_equal(values, 0.5)


def test_uniform_vector_field_returns_the_same_vector_everywhere() -> None:
    field = UniformVectorField(np.array([1.0, -2.0, 0.0]))
    vectors = field.sample(POSITIONS_XYZ)
    assert vectors.shape == (3, 3)
    np.testing.assert_array_equal(vectors, np.tile([1.0, -2.0, 0.0], (3, 1)))


@pytest.mark.parametrize(
    ("bearing_rad", "expected_xyz"),
    [
        (0.0, [4.0, 0.0, 0.0]),
        (np.pi / 2.0, [0.0, 4.0, 0.0]),
        (np.pi, [-4.0, 0.0, 0.0]),
    ],
)
def test_constant_wind_field_components_match_speed_and_bearing(
    bearing_rad: float, expected_xyz: list[float]
) -> None:
    field = ConstantWindField.from_speed_and_bearing(4.0, bearing_rad)
    np.testing.assert_allclose(
        field.sample(POSITIONS_XYZ), np.tile(expected_xyz, (3, 1)), atol=1e-12
    )


def test_fields_satisfy_their_protocols() -> None:
    assert isinstance(UniformScalarField(1.0), ScalarFieldProtocol)
    assert isinstance(UniformVectorField(np.zeros(3)), VectorFieldProtocol)
    assert isinstance(
        ConstantWindField.from_speed_and_bearing(1.0, 0.0), VectorFieldProtocol
    )
    assert isinstance(
        PatchyDensityField(np.array([[0.0, 0.0]]), 1.0, 1.0, 0.0), ScalarFieldProtocol
    )


def test_patchy_density_field_peaks_at_a_center_and_decays_to_background() -> None:
    field = PatchyDensityField(
        patch_centers_xy=np.array([[10.0, 10.0]]),
        patch_peak_density_per_m2=0.2,
        patch_radius_m=2.0,
        background_density_per_m2=0.01,
    )
    values = field.sample(
        np.array([[10.0, 10.0, 0.0], [1000.0, 1000.0, 0.0]], dtype=np.float64)
    )
    assert values[0] == pytest.approx(0.2)
    assert values[1] == pytest.approx(0.01)


def test_patchy_density_field_rejects_empty_centers() -> None:
    with pytest.raises(ValueError, match="at least one patch center"):
        PatchyDensityField(np.zeros((0, 2)), 1.0, 1.0, 0.0)


def test_random_tree_placement_field_matches_brute_force_summation() -> None:
    influence_radius_m = 2.5
    tree_fuel_load_kg_per_m2 = 0.25
    field = RandomTreePlacementField(
        extent_x_m=50.0,
        extent_y_m=50.0,
        tree_density_per_m2=0.2,
        tree_fuel_load_kg_per_m2=tree_fuel_load_kg_per_m2,
        influence_radius_m=influence_radius_m,
        seed=1,
    )
    trees_xy = field._tree_positions_xy
    positions_xyz = np.column_stack(
        [
            np.linspace(0.0, 50.0, 200),
            np.linspace(50.0, 0.0, 200),
            np.zeros(200),
        ]
    )
    squared_distance_m2 = np.sum(
        (positions_xyz[:, None, :2] - trees_xy[None, :, :]) ** 2, axis=2
    )
    contribution = np.exp(-0.5 * squared_distance_m2 / influence_radius_m**2)
    contribution[squared_distance_m2 > (3.0 * influence_radius_m) ** 2] = 0.0
    expected = tree_fuel_load_kg_per_m2 * contribution.sum(axis=1)
    np.testing.assert_allclose(field.sample(positions_xyz), expected, atol=1e-12)


def test_random_tree_placement_field_thinning_reduces_tree_count() -> None:
    dense = RandomTreePlacementField(50.0, 50.0, 0.2, 0.25, 2.5, seed=3)
    thinned = RandomTreePlacementField(
        50.0,
        50.0,
        0.2,
        0.25,
        2.5,
        seed=3,
        density_field=UniformScalarField(0.02),
    )
    assert 0 < thinned.tree_count < dense.tree_count


def test_random_tree_placement_field_rejects_zero_density_with_a_density_field() -> (
    None
):
    with pytest.raises(ValueError, match="must be positive when a density field"):
        RandomTreePlacementField(
            50.0, 50.0, 0.0, 0.25, 2.5, density_field=UniformScalarField(0.1)
        )


def test_random_tree_placement_field_rejects_wrong_position_shape() -> None:
    field = RandomTreePlacementField(50.0, 50.0, 0.1, 0.25, 2.5, seed=1)
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        field.sample(np.zeros((4, 2)))
