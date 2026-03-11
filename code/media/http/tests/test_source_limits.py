"""
test_source_limits.py — Unit tests for per-source candidate limit behaviour in downloader.py

Tests verify that:
  1. sources with candidates_images=0 are skipped in fetch_images
  2. sources with candidates_videos=0 are skipped in fetch_videos
  3. results from multiple sources are all returned (no first-source monopoly)
  4. max_candidates_per_item cap is respected
  5. already-cached files are reused without downloading
  6. label parameter is passed through to _run_download_tasks
  7. wikimedia video search is called when candidates_videos > 0

Run with:
    python3 -m pytest code/media/http/tests/test_source_limits.py -v
"""

from __future__ import annotations

import sys
import types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: make the media/http package importable without installing it
# ---------------------------------------------------------------------------

_HTTP_DIR = Path(__file__).parent.parent.resolve()
if str(_HTTP_DIR) not in sys.path:
    sys.path.insert(0, str(_HTTP_DIR))

# Stub heavy dependencies before importing downloader
for _mod in ("requests", "open_clip", "torch", "PIL", "PIL.Image", "cv2", "numpy"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import downloader  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(
    sources: list[str],
    source_limits: dict | None = None,
    max_candidates: int = 300,
) -> dict:
    return {
        "sources": sources,
        "backoff_seconds": [],
        "rate_limits": {},
        "source_limits": source_limits or {},
        "max_candidates_per_item": max_candidates,
        "download_workers": 4,
    }


def _fake_task(dest: Path, url: str = "http://example.com/img.jpg") -> dict:
    return {"dest": dest, "url": url, "info": {"source_site": "test"}}


# ---------------------------------------------------------------------------
# 1. sources with candidates_images=0 are skipped
# ---------------------------------------------------------------------------

def test_fetch_images_skips_source_with_zero_candidates(tmp_path):
    """fetch_images must not call _pexels_search_images when candidates_images=0."""
    cfg = _make_cfg(
        sources=["pexels", "pixabay"],
        source_limits={
            "pexels":  {"candidates_images": 0},
            "pixabay": {"candidates_images": 30},
        },
    )
    api_keys = {"pexels": "pk", "pixabay": "pb"}

    with ExitStack() as stack:
        mock_pexels  = stack.enter_context(patch.object(downloader, "_pexels_search_images",  return_value=[]))
        mock_pixabay = stack.enter_context(patch.object(downloader, "_pixabay_search_images", return_value=[]))
        stack.enter_context(patch.object(downloader, "_run_download_tasks", return_value=([], {}, [])))

        downloader.fetch_images("test query", 30, tmp_path, api_keys, cfg)

    mock_pexels.assert_not_called()
    mock_pixabay.assert_called_once()


# ---------------------------------------------------------------------------
# 2. sources with candidates_videos=0 are skipped
# ---------------------------------------------------------------------------

def test_fetch_videos_skips_source_with_zero_candidates(tmp_path):
    """fetch_videos must not call _pexels_search_videos when candidates_videos=0."""
    cfg = _make_cfg(
        sources=["pexels", "pixabay"],
        source_limits={
            "pexels":  {"candidates_videos": 0},
            "pixabay": {"candidates_videos": 30},
        },
    )
    api_keys = {"pexels": "pk", "pixabay": "pb"}

    with ExitStack() as stack:
        mock_pexels  = stack.enter_context(patch.object(downloader, "_pexels_search_videos",  return_value=[]))
        mock_pixabay = stack.enter_context(patch.object(downloader, "_pixabay_search_videos", return_value=[]))
        stack.enter_context(patch.object(downloader, "_run_download_tasks", return_value=([], {}, [])))

        downloader.fetch_videos("test query", 30, tmp_path, api_keys, cfg)

    mock_pexels.assert_not_called()
    mock_pixabay.assert_called_once()


# ---------------------------------------------------------------------------
# 3. results from multiple sources all returned (no first-source monopoly)
# ---------------------------------------------------------------------------

def test_fetch_images_returns_results_from_all_sources(tmp_path):
    """Results from pexels AND pixabay must both appear in the returned list."""
    cfg = _make_cfg(
        sources=["pexels", "pixabay"],
        source_limits={
            "pexels":  {"candidates_images": 5},
            "pixabay": {"candidates_images": 5},
        },
        max_candidates=300,
    )
    api_keys = {"pexels": "pk", "pixabay": "pb"}

    pexels_results  = [(tmp_path / f"pexels_{i}.jpg",  {"source_site": "pexels"})  for i in range(5)]
    pixabay_results = [(tmp_path / f"pixabay_{i}.jpg", {"source_site": "pixabay"}) for i in range(5)]

    call_count = {"n": 0}
    def _fake_run_download_tasks(tasks, cfg, max_workers=None, label=""):
        call_count["n"] += 1
        if "pexels" in label:
            return pexels_results[:len(tasks)], {}, []
        return pixabay_results[:len(tasks)], {}, []

    pexels_hits  = [{"id": i, "src": {"large": f"http://p.com/{i}.jpg"}, "url": f"http://p.com/{i}"} for i in range(5)]
    pixabay_hits = [{"id": i, "largeImageURL": f"http://pb.com/{i}.jpg",
                     "webformatURL": f"http://pb.com/{i}_s.jpg",
                     "pageURL": f"http://pb.com/photo-{i}/",
                     "tags": "nature", "user": "u", "imageWidth": 800, "imageHeight": 600} for i in range(5)]

    with ExitStack() as stack:
        stack.enter_context(patch.object(downloader, "_pexels_search_images",  return_value=pexels_hits))
        stack.enter_context(patch.object(downloader, "_pixabay_search_images", return_value=pixabay_hits))
        stack.enter_context(patch.object(downloader, "_run_download_tasks",    side_effect=_fake_run_download_tasks))
        stack.enter_context(patch.object(downloader, "_jitter"))

        result, _pending = downloader.fetch_images("nature", 5, tmp_path, api_keys, cfg)

    sources_in_result = {info["source_site"] for _, info in result}
    assert "pexels"  in sources_in_result, "pexels results must be included"
    assert "pixabay" in sources_in_result, "pixabay results must be included"


# ---------------------------------------------------------------------------
# 4. max_candidates_per_item cap is respected
# ---------------------------------------------------------------------------

def test_fetch_images_respects_max_candidates_cap(tmp_path):
    """fetch_images must not return more than max_candidates_per_item results."""
    cfg = _make_cfg(
        sources=["pexels", "pixabay"],
        source_limits={
            "pexels":  {"candidates_images": 10},
            "pixabay": {"candidates_images": 10},
        },
        max_candidates=8,  # cap at 8
    )
    api_keys = {"pexels": "pk", "pixabay": "pb"}

    def _fake_run(tasks, cfg, max_workers=None, label=""):
        return [(tmp_path / f"img_{label}_{i}.jpg", {"source_site": label}) for i in range(len(tasks))], {}, []

    pexels_hits  = [{"id": i, "src": {"large": f"http://p.com/{i}.jpg"}, "url": f"http://p.com/{i}"} for i in range(10)]
    pixabay_hits = [{"id": i, "largeImageURL": f"http://pb.com/{i}.jpg",
                     "webformatURL": f"http://pb.com/{i}_s.jpg",
                     "pageURL": f"http://pb.com/photo-{i}/",
                     "tags": "nature", "user": "u", "imageWidth": 800, "imageHeight": 600} for i in range(10)]

    with ExitStack() as stack:
        stack.enter_context(patch.object(downloader, "_pexels_search_images",  return_value=pexels_hits))
        stack.enter_context(patch.object(downloader, "_pixabay_search_images", return_value=pixabay_hits))
        stack.enter_context(patch.object(downloader, "_run_download_tasks",    side_effect=_fake_run))
        stack.enter_context(patch.object(downloader, "_jitter"))

        result, _pending = downloader.fetch_images("nature", 10, tmp_path, api_keys, cfg)

    assert len(result) <= 8, f"Expected at most 8 results, got {len(result)}"


# ---------------------------------------------------------------------------
# 5. already-cached files are reused without downloading
# ---------------------------------------------------------------------------

def test_run_download_tasks_skips_existing_files(tmp_path):
    """_run_download_tasks must reuse files that already exist on disk."""
    existing = tmp_path / "existing.jpg"
    existing.write_bytes(b"fake image data")

    missing  = tmp_path / "missing.jpg"

    cfg = {"download_workers": 2}
    tasks = [
        {"dest": existing, "url": "http://example.com/a.jpg", "info": {"source_site": "test_a"}},
        {"dest": missing,  "url": "http://example.com/b.jpg", "info": {"source_site": "test_b"}},
    ]

    downloaded_urls: list[str] = []

    def _fake_download(url, dest, headers=None, cfg=None, source="", media_type="image"):
        downloaded_urls.append(url)
        dest.write_bytes(b"downloaded")
        return None

    with patch.object(downloader, "_download_file", side_effect=_fake_download):
        with patch.object(downloader, "_write_info_sidecar"):
            saved, _, _pending = downloader._run_download_tasks(tasks, cfg, label="test")

    assert len(downloaded_urls) == 1, "Only the missing file should be downloaded"
    assert downloaded_urls[0] == "http://example.com/b.jpg"
    assert len(saved) == 2, "Both files (cached + downloaded) should be in saved"


# ---------------------------------------------------------------------------
# 6. label parameter is forwarded to _run_download_tasks
# ---------------------------------------------------------------------------

def test_fetch_images_passes_label_to_run_download_tasks(tmp_path):
    """_run_download_tasks must be called with label='img/<source>'."""
    cfg = _make_cfg(
        sources=["pexels"],
        source_limits={"pexels": {"candidates_images": 3}},
    )
    api_keys = {"pexels": "pk"}

    pexels_hits = [{"id": i, "src": {"large": f"http://p.com/{i}.jpg"}, "url": f"http://p.com/{i}"} for i in range(3)]

    labels_seen: list[str] = []
    def _fake_run(tasks, cfg, max_workers=None, label=""):
        labels_seen.append(label)
        return [], {}, []

    with ExitStack() as stack:
        stack.enter_context(patch.object(downloader, "_pexels_search_images", return_value=pexels_hits))
        stack.enter_context(patch.object(downloader, "_run_download_tasks",   side_effect=_fake_run))
        stack.enter_context(patch.object(downloader, "_jitter"))
        downloader.fetch_images("nature", 3, tmp_path, api_keys, cfg)

    assert any("img/pexels" in lbl for lbl in labels_seen), \
        f"Expected label containing 'img/pexels', got: {labels_seen}"


# ---------------------------------------------------------------------------
# 7. wikimedia video search called when candidates_videos > 0
# ---------------------------------------------------------------------------

def test_fetch_videos_calls_wikimedia_when_enabled(tmp_path):
    """_source_search_wikimedia_videos must be called when candidates_videos > 0."""
    cfg = _make_cfg(
        sources=["wikimedia"],
        source_limits={"wikimedia": {"candidates_videos": 5}},
    )
    api_keys = {}

    with ExitStack() as stack:
        mock_wm_vid = stack.enter_context(
            patch.object(downloader, "_source_search_wikimedia_videos", return_value=[]))
        stack.enter_context(patch.object(downloader, "_run_download_tasks", return_value=([], {}, [])))
        stack.enter_context(patch.object(downloader, "_jitter"))

        downloader.fetch_videos("ancient ruins", 5, tmp_path, api_keys, cfg)

    mock_wm_vid.assert_called_once()
