"""A vector field returning the same 3D vector everywhere, satisfying VectorFieldProtocol."""

import numpy as np
import numpy.typing as npt


class UniformVectorField:
    """Constant vector field: the day-one wind model, one direction and speed everywhere."""

    def __init__(self, vector_xyz: npt.NDArray[np.float64]) -> None:
        """Store the constant 3D vector returned at every position.

        Args:
            vector_xyz: Float64 array of shape (3,).
        """
        self._vector_xyz = np.asarray(vector_xyz, dtype=np.float64)

    def sample(self, positions_xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the field at a batch of positions.

        Args:
            positions_xyz: Float64 array of shape (n, 3).

        Returns:
            Float64 array of shape (n, 3), each row equal to the constant vector.
        """
        return np.tile(self._vector_xyz, (positions_xyz.shape[0], 1))
