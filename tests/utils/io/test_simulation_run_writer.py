from pathlib import Path

import numpy as np
import pytest

from src.utils.io.simulation_run_writer import write_simulation_run

OBSERVATION_COUNT = 3
RECEIVER_COUNT = 4
SAMPLE_COUNT = 16


def build_receiver_positions_xy_m() -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float64
    )


def build_receiver_signals() -> np.ndarray:
    return np.arange(
        OBSERVATION_COUNT * RECEIVER_COUNT * SAMPLE_COUNT, dtype=np.float32
    ).reshape(OBSERVATION_COUNT, RECEIVER_COUNT, SAMPLE_COUNT)


def build_ground_truth() -> list[dict[str, object]]:
    return [
        {
            "simulation_time_s": 60.0 * (index + 1),
            "component_count": 1,
            "component_centroids_xy_m": [[5.0 + index, 5.0]],
            "front_cell_count": 4 + index,
            "burning_cell_count": 4 + index,
            "source_count": 4 + index,
        }
        for index in range(OBSERVATION_COUNT)
    ]


def build_metadata() -> dict[str, object]:
    return {
        "time_step_s": 18.0,
        "observation_interval_s": 60.0,
        "observation_count": OBSERVATION_COUNT,
        "sample_rate_hz": 44100,
        "receiver_count": RECEIVER_COUNT,
    }


def write_default_run(run_directory: Path) -> Path:
    return write_simulation_run(
        run_directory,
        {"mesh": {"extent_x_m": 100.0}},
        build_receiver_positions_xy_m(),
        build_receiver_signals(),
        build_ground_truth(),
        build_metadata(),
    )


def test_every_contract_file_is_written(tmp_path: Path) -> None:
    run_directory = write_default_run(tmp_path / "run")
    written = sorted(path.name for path in run_directory.iterdir())
    assert written == [
        "config.json",
        "ground_truth.json",
        "metadata.json",
        "receiver_positions_xy_m.npy",
        "receiver_signals.npy",
    ]


def test_signals_are_stored_as_float32(tmp_path: Path) -> None:
    run_directory = write_default_run(tmp_path / "run")
    assert np.load(run_directory / "receiver_signals.npy").dtype == np.float32


def test_float64_signals_are_narrowed_on_the_way_out(tmp_path: Path) -> None:
    run_directory = write_simulation_run(
        tmp_path / "run",
        {},
        build_receiver_positions_xy_m(),
        build_receiver_signals().astype(np.float64),
        build_ground_truth(),
        build_metadata(),
    )
    assert np.load(run_directory / "receiver_signals.npy").dtype == np.float32


def test_positions_stay_float64(tmp_path: Path) -> None:
    run_directory = write_default_run(tmp_path / "run")
    assert np.load(run_directory / "receiver_positions_xy_m.npy").dtype == np.float64


def test_parent_directories_are_created(tmp_path: Path) -> None:
    run_directory = write_default_run(tmp_path / "deep" / "nested" / "run")
    assert run_directory.is_dir()


def test_a_two_dimensional_signal_array_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_observations, n_receivers, n_samples"):
        write_simulation_run(
            tmp_path / "run",
            {},
            build_receiver_positions_xy_m(),
            np.zeros((RECEIVER_COUNT, SAMPLE_COUNT), dtype=np.float32),
            build_ground_truth(),
            build_metadata(),
        )


def test_ground_truth_shorter_than_the_signals_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ground truth records supplied"):
        write_simulation_run(
            tmp_path / "run",
            {},
            build_receiver_positions_xy_m(),
            build_receiver_signals(),
            build_ground_truth()[:1],
            build_metadata(),
        )
