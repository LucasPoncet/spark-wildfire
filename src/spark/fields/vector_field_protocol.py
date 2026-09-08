"""Protocol satisfied by every vector spatial field in the repository.

A vector field is a function from position to a 3D vector: wind, in the
day-one usage. Consumers depend on this Protocol only, never on a
concrete implementation.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class VectorFieldProtocol(Protocol):
    """A field mapping 3D positions to a 3D vector value."""

    def sample(self, positions_xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the field at a batch of positions.

        Args:
            positions_xyz: Float64 array of shape (n, 3).

        Returns:
            Float64 array of shape (n, 3).
        """
        ...
