"""Root configuration for a fire simulation rendered to receiver signals.

Composes the six component configs into one JSON-serialisable object. This
is deliberately not `simulation_configuration.SimulationConfiguration`,
which is the TOML-loaded root of the two-receiver localization pipeline and
is depended on by `conftest.py`, four `inverse/` modules and six test files.
The two roots describe different experiments and are kept apart.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from src.config.acoustic_rendering_configuration import AcousticRenderingConfiguration
from src.config.atmospheric_configuration import AtmosphericConfiguration
from src.config.fire_simulation_configuration import FireSimulationConfiguration
from src.config.mesh_configuration import MeshConfiguration
from src.config.receiver_configuration import ReceiverConfiguration
from src.config.wind_configuration import WindConfiguration

JSON_INDENT: int = 2


@dataclass(frozen=True)
class FireRenderingConfiguration:
    """Everything one fire-to-receivers run needs, in one serialisable object.

    Attributes:
        mesh: Grid geometry.
        wind: Uniform wind field.
        fire: Fuel, ignition point and run length.
        receiver: Microphone layout.
        atmosphere: Air state for absorption.
        acoustic: Sampling and signal parameters.
    """

    mesh: MeshConfiguration = field(default_factory=MeshConfiguration)
    wind: WindConfiguration = field(default_factory=WindConfiguration)
    fire: FireSimulationConfiguration = field(
        default_factory=FireSimulationConfiguration
    )
    receiver: ReceiverConfiguration = field(default_factory=ReceiverConfiguration)
    atmosphere: AtmosphericConfiguration = field(
        default_factory=AtmosphericConfiguration
    )
    acoustic: AcousticRenderingConfiguration = field(
        default_factory=AcousticRenderingConfiguration
    )

    @classmethod
    def default(cls) -> Self:
        """Build the all-defaults configuration.

        Returns:
            The default configuration.
        """
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build from a JSON-decoded mapping, section by section.

        Absent sections and absent keys within a section both fall back to
        defaults, so a partial override leaves everything else untouched.

        Args:
            data: Mapping of section name to that section's mapping.

        Returns:
            The configuration.
        """
        return cls(
            mesh=MeshConfiguration.from_dict(data.get("mesh", {})),
            wind=WindConfiguration.from_dict(data.get("wind", {})),
            fire=FireSimulationConfiguration.from_dict(data.get("fire", {})),
            receiver=ReceiverConfiguration.from_dict(data.get("receiver", {})),
            atmosphere=AtmosphericConfiguration.from_dict(data.get("atmosphere", {})),
            acoustic=AcousticRenderingConfiguration.from_dict(data.get("acoustic", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping, section by section.

        Returns:
            Mapping of section name to that section's mapping.
        """
        return {
            "mesh": self.mesh.to_dict(),
            "wind": self.wind.to_dict(),
            "fire": self.fire.to_dict(),
            "receiver": self.receiver.to_dict(),
            "atmosphere": self.atmosphere.to_dict(),
            "acoustic": self.acoustic.to_dict(),
        }

    @classmethod
    def from_json(cls, path: Path) -> Self:
        """Read a configuration from a JSON file.

        Args:
            path: File to read.

        Returns:
            The configuration.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.is_file():
            raise FileNotFoundError(f"configuration file not found: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_json(self, path: Path) -> None:
        """Write this configuration to a JSON file, creating parent directories.

        Args:
            path: File to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=JSON_INDENT) + "\n", encoding="utf-8"
        )
