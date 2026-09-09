"""What this scene is called, so runs and figures can label themselves.

Without a name, a run directory and a figure file carry only a timestamp, and
nothing tells a reader which rung of the ladder they belong to. The name is
the one piece of a configuration that exists purely so the outputs are
self-describing.
"""

import re
from dataclasses import dataclass
from typing import Any, Self

DEFAULT_EXPERIMENT_NAME: str = "default"
SAFE_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Identity of one scene.

    Attributes:
        name: Short identifier used as a directory name for runs and figures.
            Lower case, digits, hyphen and underscore only, so it is safe on
            every filesystem.
        title: One line naming the scene for a figure title or a picker.
        description: What the scene is for, in a sentence or two.
    """

    name: str = DEFAULT_EXPERIMENT_NAME
    title: str = "Default scene"
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build from a decoded mapping, filling absent keys with defaults.

        Args:
            data: Mapping of field name to value.

        Returns:
            The configuration.

        Raises:
            ValueError: If the name is not a safe directory name.
        """
        defaults = cls()
        name = str(data.get("name", defaults.name))
        if SAFE_NAME_PATTERN.match(name) is None:
            raise ValueError(
                f"experiment name {name!r} must be lower case letters, digits, "
                "hyphen or underscore, starting with a letter or digit"
            )
        return cls(
            name=name,
            title=str(data.get("title", defaults.title)),
            description=str(data.get("description", defaults.description)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
        }


def format_scene_label(name: str, title: str) -> str:
    """Compose the label a picker shows for one scene.

    Args:
        name: Short identifier, the directory name.
        title: One line naming the scene.

    Returns:
        The label, falling back to the name alone when there is no title.
    """
    return f"{name} — {title}" if title else name
