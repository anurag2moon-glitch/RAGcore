"""Durable storage for the original, unmodified file bytes behind a
Document — see ragcore/raw_files/base.py's module docstring."""

from ragcore.raw_files.base import RawFileStore
from ragcore.raw_files.s3 import S3RawFileStore

__all__ = ["RawFileStore", "S3RawFileStore"]
