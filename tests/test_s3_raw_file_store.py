"""S3RawFileStore, exercised against moto's in-process S3 mock — real
boto3 client, real S3 wire protocol, no network and no real AWS account
(docs/ARCHITECTURE.md Phase 2 gap list item 12: raw file storage)."""

from __future__ import annotations

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from ragcore.errors import StoreError
from ragcore.raw_files.s3 import S3RawFileStore

BUCKET = "ragcore-test-bucket"


@pytest.fixture
def s3_store():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield S3RawFileStore(BUCKET, client=client)


def test_put_then_get_round_trips_the_file(tmp_path: Path, s3_store: S3RawFileStore) -> None:
    src = tmp_path / "book.pdf"
    src.write_bytes(b"pdf bytes here")

    uri = s3_store.put("source-1", src)
    assert uri == f"s3://{BUCKET}/source-1"

    dest = tmp_path / "downloaded.pdf"
    s3_store.get("source-1", dest)
    assert dest.read_bytes() == b"pdf bytes here"


def test_exists_reflects_whether_a_source_id_was_stored(
    tmp_path: Path, s3_store: S3RawFileStore
) -> None:
    assert s3_store.exists("missing") is False

    src = tmp_path / "f.txt"
    src.write_bytes(b"x")
    s3_store.put("present", src)

    assert s3_store.exists("present") is True
    assert s3_store.exists("missing") is False


def test_get_on_a_missing_source_id_raises_store_error(
    tmp_path: Path, s3_store: S3RawFileStore
) -> None:
    with pytest.raises(StoreError, match="no raw file stored"):
        s3_store.get("missing", tmp_path / "out.txt")


def test_delete_returns_true_only_when_something_was_deleted(
    tmp_path: Path, s3_store: S3RawFileStore
) -> None:
    src = tmp_path / "f.txt"
    src.write_bytes(b"x")
    s3_store.put("present", src)

    assert s3_store.delete("present") is True
    assert s3_store.exists("present") is False
    assert s3_store.delete("present") is False  # already gone
    assert s3_store.delete("never-existed") is False


def test_original_filename_round_trips_through_metadata(
    tmp_path: Path, s3_store: S3RawFileStore
) -> None:
    src = tmp_path / "The Great Gatsby.epub"
    src.write_bytes(b"epub bytes")
    s3_store.put("gatsby", src)

    assert s3_store.original_filename("gatsby") == "The Great Gatsby.epub"


def test_original_filename_is_none_for_a_missing_source_id(s3_store: S3RawFileStore) -> None:
    assert s3_store.original_filename("missing") is None


def test_different_source_ids_do_not_collide(tmp_path: Path, s3_store: S3RawFileStore) -> None:
    a = tmp_path / "a.txt"
    a.write_bytes(b"aaa")
    b = tmp_path / "b.txt"
    b.write_bytes(b"bbb")

    s3_store.put("id-a", a)
    s3_store.put("id-b", b)

    dest_a, dest_b = tmp_path / "out_a.txt", tmp_path / "out_b.txt"
    s3_store.get("id-a", dest_a)
    s3_store.get("id-b", dest_b)
    assert dest_a.read_bytes() == b"aaa"
    assert dest_b.read_bytes() == b"bbb"


def test_prefix_is_applied_to_the_s3_key(tmp_path: Path) -> None:
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        store = S3RawFileStore(BUCKET, prefix="raw/books/", client=client)

        src = tmp_path / "f.txt"
        src.write_bytes(b"x")
        uri = store.put("id-1", src)

        assert uri == f"s3://{BUCKET}/raw/books/id-1"
        # confirm the key really is prefixed, not just the returned string
        client.head_object(Bucket=BUCKET, Key="raw/books/id-1")


def test_missing_extra_raises_without_a_client() -> None:
    import ragcore.raw_files.s3 as s3_module

    original = s3_module.boto3
    s3_module.boto3 = None
    try:
        with pytest.raises(StoreError, match=r"ragcore\[s3\]"):
            S3RawFileStore(BUCKET)
    finally:
        s3_module.boto3 = original
