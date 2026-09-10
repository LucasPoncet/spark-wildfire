"""The sources a multi-source scene puts in the domain."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.array_types import Float64Array


@dataclass(frozen=True)
class SourceConfiguration:
    """One concurrent source: where it is and how loud it is.

    Attributes:
        position_xy_m: Ground-plane position `(x, y)` in metres.
        amplitude_scale: Multiplier on the emitted waveform, so a scene can put
            a quiet source next to a loud one.
    """

    position_xy_m: Float64Array
    amplitude_scale: float

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "position_xy_m": np.asarray(self.position_xy_m, dtype=np.float64).tolist(),
            "amplitude_scale": self.amplitude_scale,
        }


def stack_source_positions_xy_m(
    sources: tuple[SourceConfiguration, ...],
) -> Float64Array:
    """Collects every source position into one array.

    Args:
        sources: The scene's sources.

    Returns:
        Positions of shape `(n_sources, 2)`.

    Raises:
        ValueError: If the scene holds no source.
    """
    if not sources:
        raise ValueError("a multi-source scene must declare at least one source")
    return np.stack(
        [np.asarray(source.position_xy_m, dtype=np.float64) for source in sources]
    )


def collect_source_amplitude_scales(
    sources: tuple[SourceConfiguration, ...],
) -> Float64Array:
    """Collects every source amplitude scale into one array.

    Args:
        sources: The scene's sources.

    Returns:
        Scales of shape `(n_sources,)`.
    """
    return np.array([source.amplitude_scale for source in sources], dtype=np.float64)
