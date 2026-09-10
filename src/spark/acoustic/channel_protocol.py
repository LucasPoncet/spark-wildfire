from typing import Protocol, runtime_checkable

from src.utils.array_types import Float64Array


@runtime_checkable
class ChannelProtocol(Protocol):
    """Propagation channel between source positions and receiver positions."""

    def compute_gain_matrix(
        self,
        source_positions_xyz: Float64Array,
        receiver_positions_xyz: Float64Array,
    ) -> Float64Array:
        """Computes the amplitude gain from every source to every receiver.

        Args:
            source_positions_xyz: Source positions, shape `(n_sources, 3)`.
            receiver_positions_xyz: Receiver positions, shape `(n_receivers, 3)`.

        Returns:
            Gain matrix of shape `(n_sources, n_receivers)`. The forward render is
            `received_levels = gain_matrix.T @ source_amplitudes`.
        """
        ...
