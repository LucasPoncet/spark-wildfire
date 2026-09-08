from pathlib import Path

import numpy as np
import soundfile as sf

from src.utils.array_types import Float64Array


def read_audio_file(path: Path | str, as_mono: bool = True) -> tuple[Float64Array, int]:
    samples, sample_rate_hz = sf.read(str(path), dtype="float64", always_2d=True)
    channels = np.asarray(samples, dtype=np.float64)
    if as_mono:
        return channels.mean(axis=1), int(sample_rate_hz)
    return channels.T, int(sample_rate_hz)


def read_audio_metadata(path: Path | str) -> tuple[int, int, float]:
    info = sf.info(str(path))
    return int(info.samplerate), int(info.channels), float(info.duration)
