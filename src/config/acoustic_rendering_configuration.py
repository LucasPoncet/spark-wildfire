"""Sampling and signal parameters of the acoustic render."""

from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True)
class AcousticRenderingConfiguration:
    """How each burning cell's signal is generated, propagated and sampled.

    Attributes:
        sample_rate_hz: Audio sample rate, in hertz.
        segment_duration_s: Length of the signal rendered per observation, in
            seconds.
        reference_distance_m: Distance at which geometric spreading gain is unity.
        source_signal_seed: Base seed for per-cell source noise.
        observation_interval_s: Simulated seconds between two acoustic
            observations. Independent of the fire timestep, which physics
            bounds; how often the array listens is an experimental choice.
        channel_name: Registered name of the propagation channel.
        attenuation_coefficient_per_m: Scalar absorption used by the
            exponential channel, in inverse metres.
    """

    sample_rate_hz: int = 44100
    segment_duration_s: float = 5.0
    reference_distance_m: float = 1.0
    source_signal_seed: int = 42
    observation_interval_s: float = 60.0
    channel_name: str = "exponential_attenuation"
    attenuation_coefficient_per_m: float = 0.005

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build from a decoded mapping, filling absent keys with defaults.

        Args:
            data: Mapping of field name to value.

        Returns:
            The configuration.

        Raises:
            ValueError: If the observation interval is not positive.
        """
        defaults = cls()
        observation_interval_s = float(
            data.get("observation_interval_s", defaults.observation_interval_s)
        )
        if observation_interval_s <= 0.0:
            raise ValueError("observation interval must be positive")
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
            observation_interval_s=observation_interval_s,
            channel_name=str(data.get("channel_name", defaults.channel_name)),
            attenuation_coefficient_per_m=float(
                data.get(
                    "attenuation_coefficient_per_m",
                    defaults.attenuation_coefficient_per_m,
                )
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
            "observation_interval_s": self.observation_interval_s,
            "channel_name": self.channel_name,
            "attenuation_coefficient_per_m": self.attenuation_coefficient_per_m,
        }


def compute_observation_stride(
    observation_interval_s: float, time_step_s: float
) -> int:
    """Number of fire steps between two acoustic observations.

    Args:
        observation_interval_s: Requested simulated seconds between observations.
        time_step_s: The fire timestep, in seconds.

    Returns:
        How many fire steps to advance between renders, at least one.

    Raises:
        ValueError: If the requested interval is shorter than the fire
            timestep, which cannot be honoured and would silently degrade
            into rendering every step.
    """
    if observation_interval_s < time_step_s:
        raise ValueError(
            f"observation interval {observation_interval_s} s is shorter than the "
            f"fire timestep {time_step_s} s; the array cannot listen faster than "
            f"the simulation advances"
        )
    return max(1, round(observation_interval_s / time_step_s))
