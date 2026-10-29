import pytest

from physioload.data import SampleRecord, assert_no_group_leakage, group_split


def record(subject: str, index: int) -> SampleRecord:
    return SampleRecord(str(index), "demo", subject, f"trial-{index}", "cam-1", f"{index}.png")


def test_group_split_has_no_subject_leakage():
    rows = [record(f"s{subject}", subject * 2 + frame) for subject in range(10) for frame in range(2)]
    split = group_split(rows)
    assert_no_group_leakage(split)
    assert sum(map(len, split.values())) == len(rows)


def test_leakage_check_rejects_overlap():
    row = record("same", 1)
    with pytest.raises(ValueError):
        assert_no_group_leakage({"train": [row], "validation": [row], "test": []})

