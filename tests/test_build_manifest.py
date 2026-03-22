from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.build_manifest import (
    extract_hb_grade,
    extract_pose_index,
    load_patient_metadata,
    merge_manifest_with_metadata,
    parse_file_record,
)


def test_extract_pose_index_supports_jpg_and_jpeg_case_insensitive() -> None:
    assert extract_pose_index("Normal4_8.JPG") == 8
    assert extract_pose_index("abc_12.jpeg") == 12
    assert extract_pose_index("video.mp4") is None


def test_extract_hb_grade_maps_expected_labels() -> None:
    assert extract_hb_grade("Normal") == 1
    assert extract_hb_grade("NearNormalFlaccid") == 2
    assert extract_hb_grade("MildSpasm") == 3
    assert extract_hb_grade("ModerateFlaccid") == 4
    assert extract_hb_grade("SevereFlaccid") == 5
    assert extract_hb_grade("CompleteFlaccid") == 6
    assert extract_hb_grade("UnknownSeverity") is None


def test_parse_file_record_for_normal_layout(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    file_path = raw_dir / "Normal" / "Normal4" / "Normal4_1.JPG"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("x")

    record = parse_file_record(file_path, raw_dir)

    assert record is not None
    assert record["patient_id"] == "Normal4"
    assert record["cohort"] == "Normal"
    assert record["severity_folder"] == "Normal"
    assert record["modality"] == "image"
    assert record["pose_index"] == 1


def test_parse_file_record_for_cohort_layout_video(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    file_path = raw_dir / "Flaccid" / "ModerateFlaccid" / "ModerateFlaccid2" / "ModerateFlaccid2.mp4"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("x")

    record = parse_file_record(file_path, raw_dir)

    assert record is not None
    assert record["patient_id"] == "ModerateFlaccid2"
    assert record["cohort"] == "Flaccid"
    assert record["severity_folder"] == "ModerateFlaccid"
    assert record["modality"] == "video"
    assert record["pose_index"] is None


def test_parse_file_record_returns_none_for_unexpected_layout(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    file_path = raw_dir / "orphan.jpg"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("x")

    assert parse_file_record(file_path, raw_dir) is None


def test_merge_manifest_with_metadata_keeps_schema_and_left_join() -> None:
    file_manifest = pd.DataFrame(
        [
            {
                "patient_id": "A1",
                "cohort": "Normal",
                "severity_folder": "Normal",
                "pose_index": 1,
                "modality": "image",
                "filepath": "/tmp/A1_1.jpg",
                "hb_grade": 1,
            },
            {
                "patient_id": "B2",
                "cohort": "Flaccid",
                "severity_folder": "ModerateFlaccid",
                "pose_index": None,
                "modality": "video",
                "filepath": "/tmp/B2.mp4",
                "hb_grade": 4,
            },
        ]
    )
    meta = pd.DataFrame(
        [
            {"patient_id": "A1", "Side": "L", "Gender": "F", "Age": 22},
        ]
    )

    merged = merge_manifest_with_metadata(file_manifest, meta)

    assert list(merged.columns) == [
        "patient_id",
        "cohort",
        "severity_folder",
        "pose_index",
        "modality",
        "filepath",
        "hb_grade",
        "Side",
        "Gender",
        "Age",
    ]
    assert len(merged) == 2
    assert merged.loc[merged["patient_id"] == "A1", "Side"].iloc[0] == "L"
    assert pd.isna(merged.loc[merged["patient_id"] == "B2", "Side"]).iloc[0]


def test_load_patient_metadata_builds_patient_id_and_selects_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_meta = pd.DataFrame(
        [
            {
                "Sub-category": "Moderate",
                "Category": "Flaccid",
                "#": "2",
                "Side": "R",
                "Gender": "M",
                "Age": 44,
                "extra": "ignore",
            }
        ]
    )

    def fake_read_excel(_: Path) -> pd.DataFrame:
        return fake_meta

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    result = load_patient_metadata(Path("dummy.xlsx"))

    assert list(result.columns) == ["patient_id", "Side", "Gender", "Age"]
    assert result.iloc[0]["patient_id"] == "ModerateFlaccid2"
    assert result.iloc[0]["Age"] == 44


def test_load_patient_metadata_raises_if_columns_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pd, "read_excel", lambda _: pd.DataFrame({"Side": ["L"]}))

    with pytest.raises(ValueError, match="missing required columns"):
        load_patient_metadata(Path("dummy.xlsx"))
