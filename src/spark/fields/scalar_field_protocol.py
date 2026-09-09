"""Protocol satisfied by every scalar spatial field in the repository.

A scalar field is a function from position to a single float per position:
elevation, fuel density, attenuation coefficient, and so on. Consumers
(the mesh constructor, the fire engine, the acoustic channel) depend on
this Protocol only, never on a concrete implementation.
"""

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class ScalarFieldProtocol(Protocol):
    """A field mapping 3D positions to a scalar value."""

    def sample(self, positions_xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the field at a batch of positions.

        Args:
            positions_xyz: Float64 array of shape (n, 3).

        Returns:
            Float64 array of shape (n,).
        """
        ...
