from pathlib import Path

from pipelines.fetch_glorys12v1_jja import (
    BOUNDS,
    DATASET_ID,
    VARIABLES,
    yearly_request,
)


def test_yearly_request_is_jja_only_and_scoped(tmp_path: Path):
    command = yearly_request(2022, tmp_path)
    text = " ".join(command)
    assert DATASET_ID in command
    assert "2022-06-01T00:00:00" in command
    assert "2022-08-31T23:59:59" in command
    assert "2022-09" not in text
    assert str(BOUNDS["minimum_longitude"]) in command
    assert str(BOUNDS["maximum_depth"]) in command
    for variable in VARIABLES:
        assert variable in command
    assert "--skip-existing" in command
    assert "--netcdf-compression-level" in command


def test_dry_run_does_not_change_spatial_request(tmp_path: Path):
    normal = yearly_request(2020, tmp_path)
    dry = yearly_request(2020, tmp_path, dry_run=True)
    assert dry[:len(normal)] == normal
    assert "--dry-run" in dry

