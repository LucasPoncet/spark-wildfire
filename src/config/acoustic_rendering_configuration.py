"""Sampling and signal parameters of the acoustic render."""

from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True)
class AcousticRenderingConfiguration:
    """How each burning cell's signal is generated and propagated.

    Attributes:
        sample_rate_hz: Audio sample rate, in hertz.
        segment_duration_s: Length of the signal rendered per timestep, in seconds.
        reference_distance_m: Distance at which geometric spreading gain is unity.
        source_signal_seed: Base seed for per-cell source noise.
    """

    sample_rate_hz: int = 44100
    segment_duration_s: float = 5.0
    reference_distance_m: float = 1.0
    source_signal_seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build from a JSON-decoded mapping, filling absent keys with defaults.

        Args:
            data: Mapping of field name to value.

        Returns:
            The configuration.
        """
        defaults = cls()
        return cls(
            sample_rate_hz=int(data.get("sample_rate_hz", defaults.sample_rate_hz)),
            segment_duration_s=float(
                data.get("segment_duration_s", defaults.segment_duration_s)
            ),
            reference_distance_m=float(
                data.get("reference_distance_m", defaults.reference_distance_m)
            ),
            source_signal_seed=int(
                data.get("source_signal_seed", defaults.source_signal_seed)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "segment_duration_s": self.segment_duration_s,
            "reference_distance_m": self.reference_distance_m,
            "source_signal_seed": self.source_signal_seed,
        }
