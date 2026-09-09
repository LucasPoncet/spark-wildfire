import numpy as np

from src.utils.array_types import Float64Array


def segment_recording(
    samples: Float64Array,
    sample_rate_hz: int,
    segment_duration_s: float,
    overlap_fraction: float,
) -> Float64Array:
    """Cuts a recording into fixed-length segments.

    Args:
        samples: Mono samples.
        sample_rate_hz: Sample rate in hertz.
        segment_duration_s: Segment length in seconds.
        overlap_fraction: Fraction of a segment shared with the previous one.

    Returns:
        Segments of shape `(n_segments, segment_sample_count)`. A trailing partial
        segment is dropped rather than zero-padded.

    Raises:
        ValueError: If `overlap_fraction` is outside `[0, 1)`, or the recording is
            shorter than one segment.
    """
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must lie in [0, 1)")
    signal = np.asarray(samples, dtype=np.float64)
    segment_sample_count = round(segment_duration_s * sample_rate_hz)
    hop_sample_count = max(round(segment_sample_count * (1.0 - overlap_fraction)), 1)
    if signal.size < segment_sample_count:
        raise ValueError("recording is shorter than one segment")
    starts = range(0, signal.size - segment_sample_count + 1, hop_sample_count)
    return np.stack([signal[start : start + segment_sample_count] for start in starts])
