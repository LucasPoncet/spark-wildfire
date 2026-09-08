import numpy as np
import pytest

from src.audio.recording_segmenter import segment_recording


def test_fifty_second_recording_splits_into_ten_clips(sample_rate_hz: int) -> None:
    clips = segment_recording(np.zeros(50 * sample_rate_hz), sample_rate_hz, 5.0, 0.0)
    assert clips.shape == (10, 5 * sample_rate_hz)


def test_half_overlap_doubles_the_clip_count_less_one(sample_rate_hz: int) -> None:
    clips = segment_recording(np.zeros(50 * sample_rate_hz), sample_rate_hz, 5.0, 0.5)
    assert clips.shape[0] == 19


def test_segments_carry_consecutive_content(sample_rate_hz: int) -> None:
    samples = np.arange(10 * sample_rate_hz, dtype=np.float64)
    clips = segment_recording(samples, sample_rate_hz, 5.0, 0.0)
    assert clips[0][0] == 0.0
    assert clips[1][0] == float(5 * sample_rate_hz)


def test_configured_segmentation_covers_the_recording(
    sample_rate_hz: int, clip_duration_s: float
) -> None:
    clips = segment_recording(
        np.zeros(50 * sample_rate_hz), sample_rate_hz, clip_duration_s, 0.0
    )
    assert clips.shape[1] == int(clip_duration_s * sample_rate_hz)


def test_recording_shorter_than_one_segment_is_rejected(sample_rate_hz: int) -> None:
    with pytest.raises(ValueError):
        segment_recording(np.zeros(sample_rate_hz), sample_rate_hz, 5.0, 0.0)


def test_invalid_overlap_is_rejected(sample_rate_hz: int) -> None:
    with pytest.raises(ValueError):
        segment_recording(np.zeros(50 * sample_rate_hz), sample_rate_hz, 5.0, 1.0)
