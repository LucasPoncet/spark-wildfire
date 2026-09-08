"""A scalar density field built from a few Gaussian density bumps over a sparse background."""

import numpy as np
import numpy.typing as npt


class PatchyDensityField:
    """Fuel density peaks around a handful of hand-placed centers, sparse elsewhere."""

    def __init__(
        self,
        patch_centers_xy: npt.NDArray[np.float64],
        patch_peak_density_per_m2: float,
        patch_radius_m: float,
        background_density_per_m2: float,
    ) -> None:
        """Store patch centers and the density profile parameters.

        Args:
            patch_centers_xy: Float64 array of shape (k, 2), one row per dense zone.
            patch_peak_density_per_m2: Tree density at the center of each patch.
            patch_radius_m: Gaussian standard deviation of each patch.
            background_density_per_m2: Density far from every patch.
        """
        if background_density_per_m2 < 0.0:
            raise ValueError("background density must be non-negative")
        if patch_peak_density_per_m2 < background_density_per_m2:
            raise ValueError("patch peak density must exceed the background density")
        if patch_radius_m <= 0.0:
            raise ValueError("patch radius must be positive")

        self._patch_centers_xy = np.asarray(patch_centers_xy, dtype=np.float64)
        self._patch_peak_density_per_m2 = patch_peak_density_per_m2
        self._patch_radius_m = patch_radius_m
        self._background_density_per_m2 = background_density_per_m2

    def sample(self, positions_xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the field at a batch of positions.

        Args:
            positions_xyz: Float64 array of shape (n, 3).

        Returns:
            Float64 array of shape (n,), the local tree density in trees/m².
        """
        positions_xy = positions_xyz[:, :2]
        displacement_xy = positions_xy[:, None, :] - self._patch_centers_xy[None, :, :]
        squared_distance_m2 = np.sum(displacement_xy**2, axis=2)
        patch_contribution = np.exp(-0.5 * squared_distance_m2 / self._patch_radius_m**2)
        peak_above_background = self._patch_peak_density_per_m2 - self._background_density_per_m2
        density_above_background = peak_above_background * np.max(patch_contribution, axis=1)
        return self._background_density_per_m2 + density_above_background