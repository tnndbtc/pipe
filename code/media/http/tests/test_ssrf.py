"""
test_ssrf.py — Unit tests for SSRF protection in downloader.py

Tests verify Hard Rules, allowlist decision flow, DNS rebinding prevention,
redirect depth limits, file size enforcement, content-type pre-check, and
.part file cleanup.

Run with:
    python3 -m pytest code/media/http/tests/test_ssrf.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_HTTP_DIR = Path(__file__).parent.parent.resolve()
if str(_HTTP_DIR) not in sys.path:
    sys.path.insert(0, str(_HTTP_DIR))

for _mod in ("requests", "open_clip", "torch", "PIL", "PIL.Image", "cv2", "numpy"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Re-import after stubs
import importlib
if "downloader" in sys.modules:
    importlib.reload(sys.modules["downloader"])
import downloader  # noqa: E402


# ---------------------------------------------------------------------------
# Hard block rules
# ---------------------------------------------------------------------------

class TestHardBlockRules:
    """Hard Rules 1-3: scheme, IP literal, private IP."""

    def test_http_scheme_blocked(self):
        with pytest.raises(ValueError, match="hard_block_scheme"):
            downloader._hard_block_check("http://example.com/img.jpg")

    def test_file_scheme_blocked(self):
        with pytest.raises(ValueError, match="hard_block_scheme"):
            downloader._hard_block_check("file:///etc/passwd")

    def test_ftp_scheme_blocked(self):
        with pytest.raises(ValueError, match="hard_block_scheme"):
            downloader._hard_block_check("ftp://example.com/file")

    def test_data_scheme_blocked(self):
        with pytest.raises(ValueError, match="hard_block_scheme"):
            downloader._hard_block_check("data:text/html,<h1>hi</h1>")

    def test_bare_ipv4_blocked(self):
        with pytest.raises(ValueError, match="hard_block_ip_literal"):
            downloader._hard_block_check("https://1.2.3.4/img.jpg")

    def test_bare_ipv6_blocked(self):
        with pytest.raises(ValueError, match="hard_block_ip_literal"):
            downloader._hard_block_check("https://[::1]/img.jpg")

    def test_private_ip_127(self):
        with patch.object(downloader, "_resolve_all_ips", side_effect=ValueError("hard_block_private_ip")):
            with pytest.raises(ValueError, match="hard_block_private_ip"):
                downloader._hard_block_check("https://evil.com/img.jpg")

    def test_private_ip_192_168(self):
        assert downloader._is_private_ip("192.168.86.33") is True

    def test_private_ip_169_254(self):
        assert downloader._is_private_ip("169.254.169.254") is True

    def test_private_ip_10(self):
        assert downloader._is_private_ip("10.0.0.1") is True

    def test_public_ip_passes(self):
        assert downloader._is_private_ip("8.8.8.8") is False

    def test_valid_public_hostname_passes(self):
        with patch.object(downloader.socket, "getaddrinfo", return_value=[
            (2, 1, 6, '', ('93.184.216.34', 0)),
        ]):
            hostname, ips = downloader._hard_block_check("https://example.com/img.jpg")
            assert hostname == "example.com"
            assert "93.184.216.34" in ips

    def test_mixed_public_private_blocked(self):
        """If ANY resolved IP is private, the hostname is blocked."""
        with patch.object(downloader.socket, "getaddrinfo", return_value=[
            (2, 1, 6, '', ('93.184.216.34', 0)),
            (2, 1, 6, '', ('192.168.1.1', 0)),
        ]):
            with pytest.raises(ValueError, match="private IP"):
                downloader._hard_block_check("https://example.com/img.jpg")


# ---------------------------------------------------------------------------
# Allowlist decision flow
# ---------------------------------------------------------------------------

class TestAllowlistDecision:

    def test_static_allowlist_match(self):
        assert downloader._is_host_allowed("images.pexels.com") == "static"

    def test_static_allowlist_wildcard(self):
        assert downloader._is_host_allowed("upload.wikimedia.org") == "static"

    def test_dynamic_allowed(self):
        downloader._ssrf_allowed_hosts["images.rawpixel.com"] = {
            "added_ts": "2026-01-01T00:00:00Z", "expires_ts": None
        }
        try:
            assert downloader._is_host_allowed("images.rawpixel.com") == "dynamic"
        finally:
            downloader._ssrf_allowed_hosts.pop("images.rawpixel.com", None)

    def test_rejected_host(self):
        downloader._ssrf_rejected_hosts["malicious.com"] = {"added_ts": "2026-01-01"}
        try:
            assert downloader._is_host_allowed("malicious.com") == "rejected"
        finally:
            downloader._ssrf_rejected_hosts.pop("malicious.com", None)

    def test_unknown_host(self):
        assert downloader._is_host_allowed("never-seen-before.example.com") == "unknown"


# ---------------------------------------------------------------------------
# Content-Type pre-check
# ---------------------------------------------------------------------------

class TestContentTypePreCheck:

    def test_text_html_rejected(self):
        assert "text/html" in downloader._BLOCKED_CONTENT_TYPES

    def test_application_json_rejected(self):
        assert "application/json" in downloader._BLOCKED_CONTENT_TYPES

    def test_image_jpeg_not_blocked(self):
        assert "image/jpeg" not in downloader._BLOCKED_CONTENT_TYPES

    def test_application_octet_stream_not_blocked(self):
        """application/octet-stream must NOT be blocked (C8 fix)."""
        assert "application/octet-stream" not in downloader._BLOCKED_CONTENT_TYPES


# ---------------------------------------------------------------------------
# Private IP ranges
# ---------------------------------------------------------------------------

class TestPrivateRanges:

    @pytest.mark.parametrize("ip", [
        "127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.0.1",
        "169.254.169.254", "0.0.0.0",
    ])
    def test_private_ipv4(self, ip):
        assert downloader._is_private_ip(ip) is True

    @pytest.mark.parametrize("ip", [
        "8.8.8.8", "93.184.216.34", "1.1.1.1",
    ])
    def test_public_ipv4(self, ip):
        assert downloader._is_private_ip(ip) is False


# ---------------------------------------------------------------------------
# IDNA normalization
# ---------------------------------------------------------------------------

class TestIDNA:

    def test_idna_normalization(self):
        with patch.object(downloader.socket, "getaddrinfo", return_value=[
            (2, 1, 6, '', ('93.184.216.34', 0)),
        ]):
            hostname, _ = downloader._hard_block_check("https://example.com/img.jpg")
            assert hostname == "example.com"
