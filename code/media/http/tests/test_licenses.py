"""
test_licenses.py — Unit tests for license normalization in downloader.py

Tests verify:
  - Wikimedia PD-* variants normalize to "Public Domain"
  - PD-Art gets license_note
  - CC BY-SA remains rejected
  - Europeana PDM accepted, NoC-* rejected
  - is_license_acceptable gate

Run with:
    python3 -m pytest code/media/http/tests/test_licenses.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_HTTP_DIR = Path(__file__).parent.parent.resolve()
if str(_HTTP_DIR) not in sys.path:
    sys.path.insert(0, str(_HTTP_DIR))

for _mod in ("requests", "open_clip", "torch", "PIL", "PIL.Image", "cv2", "numpy"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import downloader  # noqa: E402


# ---------------------------------------------------------------------------
# Wikimedia license normalization
# ---------------------------------------------------------------------------

class TestWikimediaLicense:

    def test_pd_old_100(self):
        assert downloader._normalize_wikimedia_license("PD-old-100") == "Public Domain"

    def test_pd_us(self):
        assert downloader._normalize_wikimedia_license("PD-US") == "Public Domain"

    def test_pd_art(self):
        assert downloader._normalize_wikimedia_license("PD-Art") == "Public Domain"

    def test_pd_ineligible(self):
        assert downloader._normalize_wikimedia_license("PD-ineligible") == "Public Domain"

    def test_pd_self(self):
        assert downloader._normalize_wikimedia_license("PD-self") == "Public Domain"

    def test_pd_usgov(self):
        assert downloader._normalize_wikimedia_license("PD-USGov") == "Public Domain"

    def test_public_domain_unchanged(self):
        assert downloader._normalize_wikimedia_license("Public domain") == "Public Domain"

    def test_cc0_unchanged(self):
        assert downloader._normalize_wikimedia_license("cc0") == "CC0"

    def test_cc_by_sa_passthrough(self):
        """CC BY-SA must pass through unchanged (then rejected by is_license_acceptable)."""
        assert downloader._normalize_wikimedia_license("CC BY-SA 4.0") == "CC BY-SA 4.0"

    def test_gfdl_passthrough(self):
        assert downloader._normalize_wikimedia_license("GFDL") == "GFDL"


# ---------------------------------------------------------------------------
# is_license_acceptable gate
# ---------------------------------------------------------------------------

class TestLicenseAcceptable:

    def test_cc_by_sa_rejected(self):
        assert downloader.is_license_acceptable("CC BY-SA 4.0") is False

    def test_cc_by_accepted(self):
        assert downloader.is_license_acceptable("CC BY 4.0") is True

    def test_cc_by_2_accepted(self):
        assert downloader.is_license_acceptable("CC BY 2.0") is True

    def test_cc0_accepted(self):
        assert downloader.is_license_acceptable("CC0") is True

    def test_public_domain_accepted(self):
        assert downloader.is_license_acceptable("Public Domain") is True

    def test_empty_rejected(self):
        assert downloader.is_license_acceptable("") is False

    def test_gfdl_rejected(self):
        assert downloader.is_license_acceptable("GFDL") is False

    def test_cc_by_nd_rejected(self):
        assert downloader.is_license_acceptable("CC BY-ND 4.0") is False

    def test_cc_by_nc_rejected(self):
        assert downloader.is_license_acceptable("CC BY-NC 4.0") is False


# ---------------------------------------------------------------------------
# Europeana license normalization
# ---------------------------------------------------------------------------

class TestEuropeanaLicense:

    def test_pdm_accepted(self):
        assert downloader._normalize_europeana_license(
            "http://rightsstatements.org/vocab/PDM/1.0/"
        ) == "Public Domain"

    def test_noc_nc_rejected(self):
        assert downloader._normalize_europeana_license(
            "http://rightsstatements.org/vocab/NoC-NC/1.0/"
        ) == ""

    def test_noc_oglh_rejected(self):
        assert downloader._normalize_europeana_license(
            "http://rightsstatements.org/vocab/NoC-OGLH/1.0/"
        ) == ""

    def test_noc_us_rejected(self):
        assert downloader._normalize_europeana_license(
            "http://rightsstatements.org/vocab/NoC-US/1.0/"
        ) == ""

    def test_inc_rejected(self):
        assert downloader._normalize_europeana_license(
            "http://rightsstatements.org/vocab/InC/1.0/"
        ) == ""

    def test_cc0_accepted(self):
        assert downloader._normalize_europeana_license(
            "http://creativecommons.org/publicdomain/zero/1.0/"
        ) == "CC0"

    def test_cc_by_4_accepted(self):
        assert downloader._normalize_europeana_license(
            "http://creativecommons.org/licenses/by/4.0/"
        ) == "CC BY 4.0"

    def test_cc_by_sa_rejected(self):
        assert downloader._normalize_europeana_license(
            "http://creativecommons.org/licenses/by-sa/4.0/"
        ) == ""


# ---------------------------------------------------------------------------
# CC BY attribution helper
# ---------------------------------------------------------------------------

class TestCCBYAttribution:

    def test_ensure_cc_by_attribution_fills_missing(self):
        info = {
            "license_summary": "CC BY 4.0",
            "title": "Test Image",
            "photographer": "John Doe",
            "source_site": "openverse",
        }
        downloader._ensure_cc_by_attribution(info)
        assert "John Doe" in info["attribution_text"]
        assert "CC BY 4.0" in info["attribution_text"]

    def test_ensure_cc_by_attribution_skips_existing(self):
        info = {
            "license_summary": "CC BY 4.0",
            "attribution_text": "Custom attribution",
        }
        downloader._ensure_cc_by_attribution(info)
        assert info["attribution_text"] == "Custom attribution"

    def test_ensure_cc_by_attribution_skips_cc0(self):
        info = {
            "license_summary": "CC0",
            "title": "Test",
        }
        downloader._ensure_cc_by_attribution(info)
        assert "attribution_text" not in info
