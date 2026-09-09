from pathlib import Path

import numpy as np
import soundfile as sf

from src.utils.array_types import Float64Array


def read_audio_file(path: Path | str, as_mono: bool = True) -> tuple[Float64Array, int]:
    """Reads an audio file without filtering, normalizing or segmenting it.

    Args:
        path: Path to a wav or flac file.
        as_mono: Whether to average the channels down to one.

    Returns:
        `(samples, sample_rate_hz)`. Samples are `(n,)` when mono, else
        `(n_channels, n)`.
    """
    samples, sample_rate_hz = sf.read(str(path), dtype="float64", always_2d=True)
    channels = np.asarray(samples, dtype=np.float64)
    if as_mono:
        return channels.mean(axis=1), int(sample_rate_hz)
    return channels.T, int(sample_rate_hz)


def read_audio_metadata(path: Path | str) -> tuple[int, int, float]:
    """Reads an audio file's header without loading its samples.

    Args:
        path: Path to a wav or flac file.

    Returns:
        `(sample_rate_hz, channel_count, duration_s)`.
    """
    info = sf.info(str(path))
    return int(info.samplerate), int(info.channels), float(info.duration)
