from pathlib import Path

import numpy as np
import pytest

from src.utils.io.simulation_run_reader import (
    list_simulation_runs,
    read_simulation_run,
)
from tests.utils.io.test_simulation_run_writer import (
    OBSERVATION_COUNT,
    RECEIVER_COUNT,
    SAMPLE_COUNT,
    build_ground_truth,
    build_metadata,
    build_receiver_positions_xy_m,
    build_receiver_signals,
    write_default_run,
)


def test_a_written_run_reads_back_identical(tmp_path: Path) -> None:
    run = read_simulation_run(write_default_run(tmp_path / "run"))
    np.testing.assert_array_equal(
        run.receiver_positions_xy_m, build_receiver_positions_xy_m()
    )
    np.testing.assert_array_equal(run.receiver_signals, build_receiver_signals())
    assert run.ground_truth == build_ground_truth()
    assert run.metadata == build_metadata()
    assert run.config == {"mesh": {"extent_x_m": 100.0}}


def test_the_run_id_is_the_directory_name(tmp_path: Path) -> None:
    run = read_simulation_run(write_default_run(tmp_path / "20260101T000000Z"))
    assert run.run_id == "20260101T000000Z"


def test_signals_are_memory_mapped_and_float32(tmp_path: Path) -> None:
    run = read_simulation_run(write_default_run(tmp_path / "run"))
    signals = run.receiver_signals
    assert isinstance(signals, np.memmap)
    assert signals.dtype == np.float32
    assert signals.shape == (OBSERVATION_COUNT, RECEIVER_COUNT, SAMPLE_COUNT)


def test_derived_counts_match_the_arrays(tmp_path: Path) -> None:
    run = read_simulation_run(write_default_run(tmp_path / "run"))
    assert run.receiver_count == RECEIVER_COUNT
    assert run.observation_count == OBSERVATION_COUNT
    np.testing.assert_allclose(run.observation_times_s, [60.0, 120.0, 180.0])


def test_a_missing_directory_raises_naming_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="simulation run directory not found"):
        read_simulation_run(tmp_path / "absent")


def test_a_missing_file_raises_naming_it(tmp_path: Path) -> None:
    run_directory = write_default_run(tmp_path / "run")
    (run_directory / "receiver_signals.npy").unlink()
    with pytest.raises(FileNotFoundError, match=r"receiver_signals\.npy"):
        read_simulation_run(run_directory)


def test_a_missing_metadata_file_raises_naming_it(tmp_path: Path) -> None:
    run_directory = write_default_run(tmp_path / "run")
    (run_directory / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"metadata\.json"):
        read_simulation_run(run_directory)


def test_runs_are_listed_newest_first(tmp_path: Path) -> None:
    for run_id in ("20260101T000000Z", "20260301T000000Z", "20260201T000000Z"):
        write_default_run(tmp_path / run_id)
    assert list_simulation_runs(tmp_path) == [
        "20260301T000000Z",
        "20260201T000000Z",
        "20260101T000000Z",
    ]


def test_a_directory_without_metadata_is_not_a_run(tmp_path: Path) -> None:
    write_default_run(tmp_path / "20260101T000000Z")
    (tmp_path / "scratch").mkdir()
    assert list_simulation_runs(tmp_path) == ["20260101T000000Z"]


def test_listing_an_absent_root_is_empty(tmp_path: Path) -> None:
    assert list_simulation_runs(tmp_path / "absent") == []


def test_a_malformed_ground_truth_document_is_rejected(tmp_path: Path) -> None:
    run_directory = write_default_run(tmp_path / "run")
    (run_directory / "ground_truth.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON array"):
        read_simulation_run(run_directory)


def test_a_malformed_metadata_document_is_rejected(tmp_path: Path) -> None:
    run_directory = write_simulation_run_with_metadata(tmp_path / "run", "[]")
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_simulation_run(run_directory)


def write_simulation_run_with_metadata(run_directory: Path, metadata_text: str) -> Path:
    from src.utils.io.simulation_run_writer import write_simulation_run

    written = write_simulation_run(
        run_directory,
        {},
        build_receiver_positions_xy_m(),
        build_receiver_signals(),
        build_ground_truth(),
        build_metadata(),
    )
    (written / "metadata.json").write_text(metadata_text, encoding="utf-8")
    return written


def test_the_run_id_carries_the_experiment_when_the_metadata_names_one(
    tmp_path: Path,
) -> None:
    run_directory = write_simulation_run_with_experiment(
        tmp_path / "e3" / "20260101T000000Z", "e3"
    )
    run = read_simulation_run(run_directory)
    assert run.run_id == "e3/20260101T000000Z"
    assert run.experiment_name == "e3"


def test_the_run_id_falls_back_to_the_directory_name(tmp_path: Path) -> None:
    run = read_simulation_run(write_default_run(tmp_path / "20260101T000000Z"))
    assert run.run_id == "20260101T000000Z"


def test_nested_runs_are_listed_by_experiment_and_stamp(tmp_path: Path) -> None:
    write_simulation_run_with_experiment(tmp_path / "e1" / "20260101T000000Z", "e1")
    write_simulation_run_with_experiment(tmp_path / "e3" / "20260301T000000Z", "e3")
    write_simulation_run_with_experiment(tmp_path / "e2" / "20260201T000000Z", "e2")
    assert list_simulation_runs(tmp_path) == [
        "e3/20260301T000000Z",
        "e2/20260201T000000Z",
        "e1/20260101T000000Z",
    ]


def test_a_listed_nested_run_id_opens_that_run(tmp_path: Path) -> None:
    write_simulation_run_with_experiment(tmp_path / "e3" / "20260301T000000Z", "e3")
    run_id = list_simulation_runs(tmp_path)[0]
    assert read_simulation_run(tmp_path / run_id).experiment_name == "e3"


def write_simulation_run_with_experiment(
    run_directory: Path, experiment_name: str
) -> Path:
    from src.utils.io.simulation_run_writer import write_simulation_run

    metadata = dict(build_metadata())
    metadata["experiment_name"] = experiment_name
    return write_simulation_run(
        run_directory,
        {},
        build_receiver_positions_xy_m(),
        build_receiver_signals(),
        build_ground_truth(),
        metadata,
    )
