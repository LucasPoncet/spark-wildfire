"""
This file owns the entire question of what fuel properties does each position in the domain have. It answers a ScalarFieldProtocol query by computing fuel load at any position from a spatial model of tree placement.

The complexity lives here, in layers you add one at a time:

Layer 1 — uniform (today)
Every cell has the same fuel load. UniformScalarField(0.5). This is not even in random_tree_placement_field.py yet, just the flat field.

Layer 2 — random Poisson placement
Trees are placed as a Poisson point process with density λ trees/m². Each tree contributes a fuel load footprint (a Gaussian kernel of radius r_m). The field sums contributions from all trees within influence distance of the query position. The seed parameter makes it reproducible.

Layer 2.5 — heterogeneous density (this file, now)
The Poisson density λ is no longer a scalar constant but a spatial density_field: ScalarFieldProtocol. Candidates are generated at the maximum density, then thinned by the local value of density_field (normalized against tree_density_per_m2). Produces zones with many trees and zones with few, without any downstream change — sample() is untouched.

Layer 3 — clustered placement
Replace the Poisson process with a Thomas cluster process (parent points Poisson, children Gaussian around each parent). Models natural forest patching. One extra parameter: cluster radius.

Layer 4 — species heterogeneity
Each tree is assigned a species from a probability distribution. Each species maps to a different FuelProperties preset. The field now returns not a scalar but a FuelProperties object — at this point it graduates from ScalarFieldProtocol to its own FuelFieldProtocol.

Layer 5 — real forest map
Load a raster of classified vegetation (IGN BD Forêt, Copernicus land cover) and map class IDs to FuelProperties presets. The field becomes VegetationRasterField, same protocol, and the fire engine sees nothing different
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spark.fields.scalar_field_protocol import ScalarFieldProtocol


class RandomTreePlacementField:
    """Scalar fuel field generated from a reproducible Poisson tree layout, optionally spatially thinned by a density field."""

    def __init__(
        self,
        extent_x_m: float,
        extent_y_m: float,
        tree_density_per_m2: float,
        tree_fuel_load: float,
        influence_radius_m: float,
        seed: int | None = None,
        density_field: ScalarFieldProtocol | None = None,
    ) -> None:
        """Generate tree positions and store the Gaussian footprint parameters.

        If density_field is None, trees are placed as a homogeneous Poisson
        process of rate tree_density_per_m2. If density_field is provided,
        tree_density_per_m2 is treated as the maximum density: candidates are
        generated at that maximum rate, then each candidate is kept with
        probability density_field(position) / tree_density_per_m2, producing
        zones with many trees and zones with few or none.
        """
        if extent_x_m <= 0.0 or extent_y_m <= 0.0:
            raise ValueError("domain extents must be positive")
        if tree_density_per_m2 < 0.0:
            raise ValueError("tree density must be non-negative")
        if tree_fuel_load < 0.0:
            raise ValueError("tree fuel load must be non-negative")
        if influence_radius_m <= 0.0:
            raise ValueError("influence radius must be positive")

        self._tree_fuel_load = tree_fuel_load
        self._influence_radius_m = influence_radius_m
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
            local_density_per_m2 = density_field.sample(candidate_positions_xyz)
            acceptance_probability = np.clip(
                local_density_per_m2 / tree_density_per_m2, 0.0, 1.0
            )
            keep_mask = generator.uniform(size=candidate_count) < acceptance_probability
            self._tree_positions_xy = candidate_positions_xy[keep_mask]

    def sample(self, positions_xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the summed Gaussian fuel footprint at 3D positions.

        The radius is used as the Gaussian standard deviation. The z coordinate
        is ignored because this is a 2D surface fuel model.
        """
        if positions_xyz.ndim != 2 or positions_xyz.shape[1] != 3:
            raise ValueError("positions_xyz must have shape (n, 3)")

        positions_xy = positions_xyz[:, :2]
        if self._tree_positions_xy.size == 0:
            return np.zeros(positions_xyz.shape[0], dtype=np.float64)

        displacement_xy = positions_xy[:, None, :] - self._tree_positions_xy[None, :, :]
        squared_distance_m2 = np.sum(displacement_xy**2, axis=2)
        influence_limit_m2 = (3.0 * self._influence_radius_m) ** 2
        contribution = np.exp(
            -0.5 * squared_distance_m2 / self._influence_radius_m**2
        )
        contribution[squared_distance_m2 > influence_limit_m2] = 0.0
        return self._tree_fuel_load * np.sum(contribution, axis=1)