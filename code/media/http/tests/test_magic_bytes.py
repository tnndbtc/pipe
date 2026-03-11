"""
test_magic_bytes.py — Unit tests for magic-bytes validation in downloader.py

Tests verify that _check_magic_bytes correctly identifies valid and invalid
file headers for images, audio, and video.

Run with:
    python3 -m pytest code/media/http/tests/test_magic_bytes.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_HTTP_DIR = Path(__file__).parent.parent.resolve()
if str(_HTTP_DIR) not in sys.path:
    sys.path.insert(0, str(_HTTP_DIR))

for _mod in ("requests", "open_clip", "torch", "PIL", "PIL.Image", "cv2", "numpy"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import downloader  # noqa: E402


# ---------------------------------------------------------------------------
# Image magic bytes
# ---------------------------------------------------------------------------

class TestImageMagicBytes:

    def test_valid_jpeg(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        downloader._check_magic_bytes(p, "image")  # should not raise

    def test_valid_png(self, tmp_path):
        p = tmp_path / "test.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        downloader._check_magic_bytes(p, "image")

    def test_valid_gif(self, tmp_path):
        p = tmp_path / "test.gif"
        p.write_bytes(b"GIF89a" + b"\x00" * 100)
        downloader._check_magic_bytes(p, "image")

    def test_valid_webp(self, tmp_path):
        p = tmp_path / "test.webp"
        p.write_bytes(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100)
        downloader._check_magic_bytes(p, "image")

    def test_html_as_image_rejected(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.write_bytes(b"<!DOCTYPE html>" + b"\x00" * 100)
        with pytest.raises(ValueError, match="invalid_signature"):
            downloader._check_magic_bytes(p, "image")

    def test_empty_file_rejected(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="invalid_signature|empty"):
            downloader._check_magic_bytes(p, "image")


# ---------------------------------------------------------------------------
# Audio magic bytes
# ---------------------------------------------------------------------------

class TestAudioMagicBytes:

    def test_valid_mp3_sync(self, tmp_path):
        p = tmp_path / "test.mp3"
        p.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 100)
        downloader._check_magic_bytes(p, "audio")

    def test_valid_mp3_id3(self, tmp_path):
        p = tmp_path / "test.mp3"
        p.write_bytes(b"ID3\x03" + b"\x00" * 100)
        downloader._check_magic_bytes(p, "audio")

    def test_valid_ogg(self, tmp_path):
        p = tmp_path / "test.ogg"
        p.write_bytes(b"OggS\x00" + b"\x00" * 100)
        downloader._check_magic_bytes(p, "audio")

    def test_valid_flac(self, tmp_path):
        p = tmp_path / "test.flac"
        p.write_bytes(b"fLaC\x00" + b"\x00" * 100)
        downloader._check_magic_bytes(p, "audio")

    def test_valid_wav(self, tmp_path):
        p = tmp_path / "test.wav"
        p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 100)
        downloader._check_magic_bytes(p, "audio")

    def test_jpeg_as_audio_rejected(self, tmp_path):
        p = tmp_path / "test.mp3"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        with pytest.raises(ValueError, match="invalid_signature"):
            downloader._check_magic_bytes(p, "audio")


# ---------------------------------------------------------------------------
# Video magic bytes (fallback — ffprobe mocked away)
# ---------------------------------------------------------------------------

class TestVideoMagicBytes:

    def test_valid_mp4_ftyp(self, tmp_path):
        p = tmp_path / "test.mp4"
        p.write_bytes(b"\x00\x00\x00\x1cftypisom" + b"\x00" * 100)
        with patch.object(downloader.subprocess, "run", side_effect=FileNotFoundError):
            downloader._check_magic_bytes(p, "video")

    def test_valid_webm(self, tmp_path):
        p = tmp_path / "test.webm"
        p.write_bytes(b"\x1aE\xdf\xa3" + b"\x00" * 100)
        with patch.object(downloader.subprocess, "run", side_effect=FileNotFoundError):
            downloader._check_magic_bytes(p, "video")

    def test_html_as_video_rejected(self, tmp_path):
        p = tmp_path / "test.mp4"
        p.write_bytes(b"<!DOCTYPE html>" + b"\x00" * 100)
        with patch.object(downloader.subprocess, "run", side_effect=FileNotFoundError):
            with pytest.raises(ValueError, match="invalid_signature"):
                downloader._check_magic_bytes(p, "video")
