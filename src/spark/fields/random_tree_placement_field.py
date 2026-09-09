"""Fuel load at any position, generated from a reproducible random tree layout.

This file owns the whole question of how much fuel sits at a position. The
extension path from a homogeneous Poisson forest to a classified vegetation
raster is the table in Architecture.md; only the layer in use lives here.

Trees are drawn once at construction and stored. `sample` sums a truncated
Gaussian footprint per tree, evaluated through a KD-tree so cost scales
with the number of trees actually within the truncation radius of a query
position, not with the product of positions and trees.
"""

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from src.spark.fields.scalar_field_protocol import ScalarFieldProtocol

FOOTPRINT_TRUNCATION_RADII = 3.0


class RandomTreePlacementField:
    """Fuel field from a Poisson tree layout, optionally thinned by a density field."""

    def __init__(
        self,
        extent_x_m: float,
        extent_y_m: float,
        tree_density_per_m2: float,
        tree_fuel_load_kg_per_m2: float,
        influence_radius_m: float,
        seed: int | None = None,
        density_field: ScalarFieldProtocol | None = None,
    ) -> None:
        """Generate tree positions and store the Gaussian footprint parameters.

        With no density field, trees are a homogeneous Poisson process of rate
        `tree_density_per_m2`. With one, that rate is the maximum: candidates
        are drawn at the maximum and each is kept with probability
        `density_field(position) / tree_density_per_m2`, producing dense and
        sparse zones. A density field exceeding the maximum is clipped, so the
        realised density there is lower than the field asks for.

        Args:
            extent_x_m: Domain extent along x, in metres.
            extent_y_m: Domain extent along y, in metres.
            tree_density_per_m2: Poisson rate, or its maximum with a density field.
            tree_fuel_load_kg_per_m2: Peak fuel load contributed by one tree.
            influence_radius_m: Gaussian standard deviation of a tree footprint.
            seed: Seed making the layout reproducible.
            density_field: Optional spatial thinning field.

        Raises:
            ValueError: If any extent, radius or load is out of range, or if a
                density field is given with a non-positive maximum density.
        """
        if extent_x_m <= 0.0 or extent_y_m <= 0.0:
            raise ValueError("domain extents must be positive")
        if tree_density_per_m2 < 0.0:
            raise ValueError("tree density must be non-negative")
        if tree_fuel_load_kg_per_m2 < 0.0:
            raise ValueError("tree fuel load must be non-negative")
        if influence_radius_m <= 0.0:
            raise ValueError("influence radius must be positive")
        if density_field is not None and tree_density_per_m2 <= 0.0:
            raise ValueError(
                "tree density must be positive when a density field is given, "
                "because it is the maximum the field is normalised against"
            )

        self._tree_fuel_load_kg_per_m2 = tree_fuel_load_kg_per_m2
        self._influence_radius_m = influence_radius_m
        self._truncation_radius_m = FOOTPRINT_TRUNCATION_RADII * influence_radius_m
        generator = np.random.default_rng(seed)
        candidate_count = generator.poisson(
            tree_density_per_m2 * extent_x_m * extent_y_m
        )
        candidate_positions_xy = generator.uniform(
            low=(0.0, 0.0),
            high=(extent_x_m, extent_y_m),
            size=(candidate_count, 2),
        )

        if density_field is None:
            self._tree_positions_xy = candidate_positions_xy
        else:
            candidate_positions_xyz = np.column_stack(
                [candidate_positions_xy, np.zeros(candidate_count)]
            )
            acceptance_probability = np.clip(
                density_field.sample(candidate_positions_xyz) / tree_density_per_m2,
                0.0,
                1.0,
            )
            is_kept = generator.uniform(size=candidate_count) < acceptance_probability
            self._tree_positions_xy = candidate_positions_xy[is_kept]

    @property
    def tree_count(self) -> int:
        """Number of trees the layout ended up with.

        Returns:
            Count of trees surviving the optional density thinning.
        """
        return int(self._tree_positions_xy.shape[0])

    def sample(self, positions_xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Sum the truncated Gaussian footprint of every nearby tree.

        The z coordinate is ignored because this is a 2D surface fuel model.

        Args:
            positions_xyz: Float64 array of shape (n, 3).

        Returns:
            Float64 array of shape (n,), fuel load in kg per square metre.

        Raises:
            ValueError: If positions_xyz is not of shape (n, 3).
        """
        if positions_xyz.ndim != 2 or positions_xyz.shape[1] != 3:
            raise ValueError("positions_xyz must have shape (n, 3)")

        position_count = positions_xyz.shape[0]
        if self._tree_positions_xy.size == 0 or position_count == 0:
            return np.zeros(position_count, dtype=np.float64)

        positions_xy = np.ascontiguousarray(positions_xyz[:, :2])
        pairs_within_radius = cKDTree(positions_xy).sparse_distance_matrix(
            cKDTree(self._tree_positions_xy),
            self._truncation_radius_m,
            output_type="coo_matrix",
        )
        contribution = np.exp(
            -0.5 * (pairs_within_radius.data / self._influence_radius_m) ** 2
        )
        return self._tree_fuel_load_kg_per_m2 * np.bincount(
            pairs_within_radius.row, weights=contribution, minlength=position_count
        )
