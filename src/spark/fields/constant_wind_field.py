"""A constant wind field expressed in physical terms (speed and bearing)."""

import numpy as np
import numpy.typing as npt

from spark.fields.uniform_vector_field import UniformVectorField


class ConstantWindField(UniformVectorField):
    """Uniform wind, constructed from a speed and a bearing rather than a raw vector."""

    @classmethod
    def from_speed_and_bearing(
        cls, wind_speed_m_per_s: float, wind_bearing_rad: float
    ) -> "ConstantWindField":
        """Build the field from a physical wind speed and bearing.

        Args:
            wind_speed_m_per_s: Wind speed, in metres per second.
            wind_bearing_rad: Direction the wind blows towards, in radians,
                measured counter-clockwise from the positive x-axis.

        Returns:
            A ConstantWindField carrying the equivalent 3D vector.
        """
        wind_vector_xyz: npt.NDArray[np.float64] = np.array(
            [
                wind_speed_m_per_s * np.cos(wind_bearing_rad),
                wind_speed_m_per_s * np.sin(wind_bearing_rad),
                0.0,
            ],
            dtype=np.float64,
        )
        return cls(wind_vector_xyz)
