"""A scalar field returning the same value everywhere.

Satisfies ScalarFieldProtocol.
"""

import numpy as np
import numpy.typing as npt


class UniformScalarField:
    """Constant scalar field: elevation, fuel density, or attenuation on day one."""

    def __init__(self, value: float) -> None:
        """Store the constant value returned at every position.

        Args:
            value: The scalar returned for any query position.
        """
        self._value = value

    def sample(self, positions_xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the field at a batch of positions.

        Args:
            positions_xyz: Float64 array of shape (n, 3).

        Returns:
            Float64 array of shape (n,), filled with the constant value.
        """
        return np.full(positions_xyz.shape[0], self._value, dtype=np.float64)
