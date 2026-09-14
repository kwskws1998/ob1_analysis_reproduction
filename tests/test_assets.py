"""Tests for byte verification, resumable downloads, and safe extraction."""

from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ob1_repro import assets


class FakeResponse(io.BytesIO):
    """Provide the context-manager surface used by urllib responses."""

    status = 206

    def __enter__(self):
        """Return this in-memory response."""
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        """Close the in-memory response."""
        self.close()


class AssetTests(unittest.TestCase):
    """Exercise download integrity and archive safety controls."""

    def test_download_resumes_and_verifies_before_install(self) -> None:
        """Resume a partial asset and require exact final bytes and digest."""
        payload = b"standalone-ob1-download-payload"
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            with mock.patch.object(assets, "ROOT", root):
                destination = root / "assets/payload.bin"
                destination.parent.mkdir(parents=True)
                partial = destination.with_suffix(".bin.part")
                partial.write_bytes(payload[:9])
                asset = {
                    "source": "https://example.invalid/payload",
                    "destination": "assets/payload.bin",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }

                def respond(request, timeout):
                    """Require the correct range and return its remaining bytes."""
                    self.assertEqual(request.get_header("Range"), "bytes=9-")
                    self.assertEqual(timeout, 180)
                    return FakeResponse(payload[9:])

                with mock.patch.object(
                    assets.urllib.request,
                    "urlopen",
                    side_effect=respond,
                ):
                    installed = assets.download_file(asset, "destination")
            self.assertEqual(installed.read_bytes(), payload)
            self.assertFalse(partial.exists())

    def test_file_checksum_rejects_corruption(self) -> None:
        """Reject any byte-size or digest mismatch."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "asset.bin"
            path.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                assets.verify_file(path, 5, hashlib.sha256(b"right").hexdigest())

    def test_tar_validation_rejects_parent_traversal(self) -> None:
        """Reject a tar member that would escape the extraction directory."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive_path = Path(temporary_dir) / "unsafe.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                payload = b"unsafe"
                member = tarfile.TarInfo("root/../../outside.txt")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            with tarfile.open(archive_path, "r:gz") as archive:
                with self.assertRaises(ValueError):
                    assets._validated_tar_root(archive)


if __name__ == "__main__":
    unittest.main()
