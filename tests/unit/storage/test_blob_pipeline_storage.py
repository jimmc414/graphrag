"""Tests for blob pipeline storage helpers."""

import pytest

from graphrag.storage.blob_pipeline_storage import validate_blob_container_name


@pytest.mark.parametrize(
    "container_name, expected_message",
    [
        ("ab", "between 3 and 63 characters"),
        ("a" * 64, "between 3 and 63 characters"),
        ("-abc", "start with a letter or number"),
        ("Abc", "must only contain"),
        ("abc--def", "consecutive hyphens"),
        ("abc-", "end with a hyphen"),
    ],
)
def test_validate_blob_container_name_invalid(container_name: str, expected_message: str) -> None:
    """Invalid container names should raise informative ValueErrors."""
    with pytest.raises(ValueError) as exc_info:
        validate_blob_container_name(container_name)

    assert expected_message in str(exc_info.value)


@pytest.mark.parametrize(
    "container_name",
    [
        "abc",
        "valid-container-1",
        "container-name-with-hyphen",
    ],
)
def test_validate_blob_container_name_valid(container_name: str) -> None:
    """Valid container names should return True."""
    assert validate_blob_container_name(container_name) is True
