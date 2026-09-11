"""How fast the front is travelling, from a series of deconvolved extents.

With `u` the bearing unit vector, `c(t)` the centroid and `a(t)` the major
semi-axis, the head is at `c(t) + a(t) u` and the back at `c(t) - a(t) u`. The
head and back rates are the slopes of those projections against time, and the
centroid speed and expansion rate are the slopes of their two ingredients.

The plan presents the closure identity — `head - back = 2 * expansion` — as a
free consistency check that catches a mis-scaled shape factor. **It does not,
and the reason is worth stating so the number is not over-read.** Head and back
are both built from the same centroid and the same semi-axis, so their
difference is twice that semi-axis whatever the bearing is and whatever the
shape factor scales it by; substituting either leaves the identity exactly
satisfied. Fitted with ordinary least squares the residual would be zero to
machine precision on any input at all.

What is left is narrower still. Theil-Sen is a median of pairwise slopes and
does not generally distribute over a sum, but it does whenever *either* addend
has constant pairwise slopes — that is, whenever either is exactly linear. Head
and back are `centroid + semi_axis` and `centroid - semi_axis`, so a straight
centroid track alone satisfies the identity exactly however the semi-axis
curves, and a straight semi-axis alone does the same however the centroid
curves. Only a run in which **both** depart from linearity leaves any residual.

Curvature alone is not enough either. Two quadratic series make every pairwise
slope a monotone function of `t_i + t_j`, so both medians fall on the same pair
and the identity survives that too. What is left is a residual that departs
from zero only when the two series disagree about which pair carries the median
slope — which takes irregularity, not shape.

So the honest reading is that this check is very nearly empty. It is kept
because it costs one subtraction and its being non-zero does say the series are
irregular, which is worth knowing when every rate quoted is a single slope. It
is not, in any of the three senses the plan claims, a check on the shape
factor, the bearing or the extent.

Slopes are Theil-Sen throughout. An extent series begins while the fire is a
few cells across, and one bad early frame under least squares would set the
rate for the whole run.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import theilslopes

from src.spark.inverse.fire_extent import ExtentEstimate, select_resolved
from src.utils.array_types import Float64Array


@dataclass(frozen=True)
class RateOfSpreadEstimate:
    """The front's speed along its own bearing, with its consistency check.

    Attributes:
        head_rate_of_spread_m_per_s: Speed of the leading edge.
        back_rate_of_spread_m_per_s: Speed of the trailing edge, negative when
            the back is retreating into burnt ground.
        centroid_speed_m_per_s: Speed of the front's centre along the bearing.
        expansion_rate_m_per_s: Rate the front's half-length grows.
        bearing_rad: Direction everything is projected onto.
        confidence_interval_m_per_s: Bootstrap interval on the head rate.
        frames_used: Resolved frames the fit ran on.
        closure_residual_m_per_s: `|head - back - 2 * expansion|`. Zero
            whenever the centroid travels at a constant speed, whatever the
            semi-axis does; non-zero only when that track curves. Not a check
            on the shape factor or the bearing, both of which leave the
            identity exactly satisfied.
        closure_residual_fraction: That residual over the head rate.
    """

    head_rate_of_spread_m_per_s: float
    back_rate_of_spread_m_per_s: float
    centroid_speed_m_per_s: float
    expansion_rate_m_per_s: float
    bearing_rad: float
    confidence_interval_m_per_s: tuple[float, float]
    frames_used: int
    closure_residual_m_per_s: float
    closure_residual_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Renders as a JSON-serialisable mapping.

        Returns:
            Mapping of field name to value.
        """
        return {
            "head_rate_of_spread_m_per_s": self.head_rate_of_spread_m_per_s,
            "back_rate_of_spread_m_per_s": self.back_rate_of_spread_m_per_s,
            "centroid_speed_m_per_s": self.centroid_speed_m_per_s,
            "expansion_rate_m_per_s": self.expansion_rate_m_per_s,
            "bearing_rad": self.bearing_rad,
            "confidence_interval_m_per_s": list(self.confidence_interval_m_per_s),
            "frames_used": self.frames_used,
            "closure_residual_m_per_s": self.closure_residual_m_per_s,
            "closure_residual_fraction": self.closure_residual_fraction,
        }


def project_head_and_back_m(
    extents: list[ExtentEstimate], bearing_rad: float
) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array, Float64Array]:
    """Projects each frame's centroid, head and back onto the bearing.

    Args:
        extents: Resolved frames, in time order.
        bearing_rad: Direction of spread.

    Returns:
        `(times_s, centroid_m, head_m, back_m, semi_axis_m)`, each of shape
        `(n_frames,)`, with the first three measured along the bearing.
    """
    direction = np.array([np.cos(bearing_rad), np.sin(bearing_rad)])
    times_s = np.array([extent.time_s for extent in extents], dtype=np.float64)
    centroid_m = np.array(
        [float(np.asarray(extent.centroid_xy_m) @ direction) for extent in extents]
    )
    semi_axis_m = np.array(
        [extent.semi_axis_major_m for extent in extents], dtype=np.float64
    )
    return (
        times_s,
        centroid_m,
        centroid_m + semi_axis_m,
        centroid_m - semi_axis_m,
        semi_axis_m,
    )


def estimate_rate_of_spread(
    extents: list[ExtentEstimate],
    bearing_rad: float,
    minimum_resolved_frames: int,
    bootstrap_resample_count: int,
    random_seed: int,
) -> RateOfSpreadEstimate:
    """Fits head, back, centroid and expansion rates against time.

    Args:
        extents: Every frame's extent; only the resolved ones are used.
        bearing_rad: Direction of spread, from the bearing stage.
        minimum_resolved_frames: Fewest resolved frames a fit may run on.
        bootstrap_resample_count: Resamples over frames for the interval.
        random_seed: Seed making the bootstrap reproducible.

    Returns:
        The rates with their closure residual.

    Raises:
        ValueError: If too few frames resolved to fit anything.
    """
    resolved = select_resolved(extents)
    if len(resolved) < minimum_resolved_frames:
        raise ValueError(
            f"a rate of spread needs at least {minimum_resolved_frames} resolved "
            f"frames, got {len(resolved)}"
        )
    times_s, centroid_m, head_m, back_m, semi_axis_m = project_head_and_back_m(
        resolved, bearing_rad
    )
    head_rate = float(theilslopes(head_m, times_s)[0])
    back_rate = float(theilslopes(back_m, times_s)[0])
    centroid_rate = float(theilslopes(centroid_m, times_s)[0])
    expansion_rate = float(theilslopes(semi_axis_m, times_s)[0])

    generator = np.random.default_rng(random_seed)
    resampled = np.empty(bootstrap_resample_count, dtype=np.float64)
    for resample_index in range(bootstrap_resample_count):
        chosen = np.sort(generator.integers(0, times_s.size, times_s.size))
        if np.unique(times_s[chosen]).size < 2:
            resampled[resample_index] = head_rate
            continue
        resampled[resample_index] = float(
            theilslopes(head_m[chosen], times_s[chosen])[0]
        )
    closure_residual = abs(head_rate - back_rate - 2.0 * expansion_rate)
    return RateOfSpreadEstimate(
        head_rate_of_spread_m_per_s=head_rate,
        back_rate_of_spread_m_per_s=back_rate,
        centroid_speed_m_per_s=centroid_rate,
        expansion_rate_m_per_s=expansion_rate,
        bearing_rad=bearing_rad,
        confidence_interval_m_per_s=(
            float(np.percentile(resampled, 2.5)),
            float(np.percentile(resampled, 97.5)),
        ),
        frames_used=len(resolved),
        closure_residual_m_per_s=closure_residual,
        closure_residual_fraction=(
            closure_residual / abs(head_rate) if abs(head_rate) > 0.0 else float("inf")
        ),
    )
