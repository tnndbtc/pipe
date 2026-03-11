"""
downloader.py — Pexels + Pixabay media search and download for code/media/http/

All public functions are *synchronous* and designed to be called via
asyncio.to_thread() from server.py so they never block the event loop.

Features
--------
- Per-provider jittered sleep after each API call (respects rate limits)
- 429 exponential backoff with configurable retry delays
- In-process search-response cache (key = source + media_type + query + per_page)
  eliminates duplicate API calls within a batch when items share similar prompts
- prefer-aware: caller passes n_img / n_vid; passing 0 skips that media type
- Query rotation: when item["search_queries"] is set, each query is issued
  independently and results are deduplicated before trimming to n
- Source filters: item["source_filters"] overrides default params per source

Public API
----------
fetch_images(search_prompt, n, output_dir, api_keys, cfg, item=None) → tuple[list[tuple[Path, dict]], list[dict]]
fetch_videos(search_prompt, n, output_dir, api_keys, cfg, item=None) → tuple[list[tuple[Path, dict]], list[dict]]

output_dir layout (created by this module):
    output_dir/
        pexels/   pexels_img_<id>.jpg   or   pexels_vid_<id>.mp4
        pixabay/  pixabay_img_<id>.<ext>  or  pixabay_vid_<id>.mp4
"""

from __future__ import annotations

import fnmatch
import ipaddress
import json
import logging
import math
import random
import re
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

import requests

log = logging.getLogger("downloader")

# Wikimedia's robot policy requires a User-Agent with contact info.
# Format: AppName/version (URL-or-email) — ref: https://w.wiki/4wJS
_USER_AGENT = "media-fetcher/1.0 (https://github.com/media-pipeline; offline-pipeline-bot)"

# ---------------------------------------------------------------------------
# Per-host global download rate limiters (token bucket)
# ---------------------------------------------------------------------------
# Controls the total request rate to a CDN hostname across ALL download
# threads in the process — not just concurrency.
#
# Example: upload.wikimedia.org at 1 req/sec means that regardless of how
# many parallel items are running, the combined throughput to that host never
# exceeds 1 request/second.  Pexels/Pixabay have no limiter — unchanged.
#
# Config (per source in rate_limits):
#   download_host           : "upload.wikimedia.org"  — CDN hostname to throttle
#   download_rate_per_sec   : 1.0                     — max global requests/sec
#   download_host_semaphore : 2                        — max concurrent in-flight
#
# The two knobs work together:
#   semaphore  — caps burst (how many requests can be in-flight simultaneously)
#   rate limit — caps throughput (how many requests per second over time)


class _HostRateLimiter:
    """Token-bucket rate limiter + concurrency semaphore for one CDN host."""

    def __init__(self, rate_per_sec: float, max_concurrent: int) -> None:
        self._interval  = 1.0 / max(rate_per_sec, 0.01)   # seconds between tokens
        self._semaphore = threading.Semaphore(max(max_concurrent, 1))
        self._lock      = threading.Lock()
        self._next_time = 0.0   # earliest time next token is available

    def acquire(self) -> None:
        """Block until both a concurrency slot and a rate token are available."""
        self._semaphore.acquire()
        with self._lock:
            now  = time.monotonic()
            wait = self._next_time - now
            if wait > 0:
                time.sleep(wait)
            # Advance the token clock; add small jitter (±10 %) to avoid bursts
            jitter = self._interval * random.uniform(-0.1, 0.1)
            self._next_time = time.monotonic() + self._interval + jitter

    def release(self) -> None:
        self._semaphore.release()


_host_limiters:     dict[str, _HostRateLimiter] = {}
_host_limiter_lock: threading.Lock              = threading.Lock()


def _get_host_limiter(hostname: str, cfg: dict) -> _HostRateLimiter | None:
    """Return the global rate limiter for hostname, or None if not configured."""
    if hostname in _host_limiters:
        return _host_limiters[hostname]
    with _host_limiter_lock:
        if hostname in _host_limiters:
            return _host_limiters[hostname]
        for src_cfg in cfg.get("rate_limits", {}).values():
            host = src_cfg.get("download_host", "")
            if host != hostname:
                continue
            rate    = float(src_cfg.get("download_rate_per_sec",   1.0))
            slots   = int(  src_cfg.get("download_host_semaphore", 2))
            limiter = _HostRateLimiter(rate, slots)
            _host_limiters[hostname] = limiter
            log.info("Created download rate limiter: host=%s rate=%.2f/s concurrent=%d",
                     hostname, rate, slots)
            return limiter
    return None


# ---------------------------------------------------------------------------
# License gate — RULE 1 + RULE 2
# ---------------------------------------------------------------------------

ACCEPTED_LICENSES: frozenset = frozenset({"CC0", "Public Domain", "CC BY", "PDM"})
ACCEPTED_PREFIXES: tuple = (
    "CC0", "Public Domain",
    "CC BY 1", "CC BY 2", "CC BY 3", "CC BY 4",
    # CC BY-SA is excluded: its ShareAlike clause conflicts with YouTube's ToS
    # (YouTube asserts rights over uploads that are incompatible with SA re-licensing).
)

def is_license_acceptable(license_summary: str) -> bool:
    """Central gate: return True only for commercially safe, derivative-friendly licenses."""
    if not license_summary:
        return False
    s = license_summary.strip()
    if s in ACCEPTED_LICENSES:
        return True
    if any(s.startswith(p) for p in ACCEPTED_PREFIXES):
        return True
    return False


ALLOWLIST_COLLECTIONS: frozenset = frozenset({
    "nasa", "metropolitanmuseumofart", "smithsonian",
})

DOWNLOAD_ALLOWLIST: list = [
    "images.pexels.com", "videos.pexels.com",
    "vod-progressive.akamaized.net",  # Pexels video CDN (Akamai)
    "cdn.pixabay.com", "pixabay.com",
    # Pixabay pre-2020 video CDN: videos.*.url in the API returns player.vimeo.com
    # URLs for older content that was originally hosted on Vimeo before Pixabay migrated
    # to their own CDN.  Without these, all such videos return "pending" and 0 are saved.
    "player.vimeo.com", "*.vimeocdn.com", "vimeocdn.com",
    "upload.wikimedia.org", "*.wikimedia.org",
    "archive.org", "ia*.us.archive.org", "ia*.archive.org",
    "api.europeana.eu", "europeanastatic.eu", "*.europeana.eu",
    # Europeana thumbnail proxy final-hop CDNs (served via api.europeana.eu/thumbnail redirect):
    "europeana-iiif.org", "*.europeana-iiif.org",
    "live.staticflickr.com", "farm*.staticflickr.com",
    "cdn.openverse.engineering", "api.openverse.org",
    "cdn.freesound.org", "freesound.org",
    "*.openverse.engineering",
]

def _check_download_allowlist(url: str) -> bool:
    try:
        hostname = urlparse(url).hostname or ""
        return any(fnmatch.fnmatch(hostname, p) for p in DOWNLOAD_ALLOWLIST)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SSRF Protection — Hard Rules + DNS Rebinding Prevention
# ---------------------------------------------------------------------------

_PRIVATE_RANGES: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

# Maximum redirect hops before hard block
_MAX_REDIRECT_DEPTH = 5

# Default rate limits for dynamically approved Tier-2 hosts (A17)
_DEFAULT_DYNAMIC_RATE_LIMIT = {
    "delay_ms_min": 300,
    "delay_ms_max": 800,
    "max_concurrent": 2,
}

# Content-Type values that are ALWAYS rejected before streaming
_BLOCKED_CONTENT_TYPES = frozenset({
    "text/html", "text/xml", "application/json", "application/javascript",
    "text/javascript", "application/xml", "text/plain",
})


def _is_private_ip(ip_str: str) -> bool:
    """Return True if ip_str falls in any private/reserved range."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable → treat as private (safe default)
    return any(addr in net for net in _PRIVATE_RANGES)


def _resolve_all_ips(hostname: str) -> list[str]:
    """Resolve ALL A + AAAA records for hostname. Raises ValueError if any is private."""
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed for {hostname}: {exc}") from exc

    ips = list({r[4][0] for r in results})
    if not ips:
        raise ValueError(f"DNS returned no records for {hostname}")

    for ip in ips:
        if _is_private_ip(ip):
            log.warning("SSRF block reason=hard_block_private_ip host=%s ip=%s", hostname, ip)
            raise ValueError(f"DNS record for {hostname} resolves to private IP {ip}")

    return ips


class _BoundHostAdapter(requests.adapters.HTTPAdapter):
    """HTTP adapter that carries pre-validated IPs for logging.

    NOTE: We intentionally do NOT rewrite the URL to a raw IP address.
    Rewriting the URL breaks TLS certificate verification because urllib3
    verifies the cert against the URL hostname (the IP), not the Host header,
    and the server's cert is for the original domain — causing an SSL mismatch
    on every request.

    DNS pre-validation in _hard_block_check (resolve all A/AAAA, reject private
    ranges) provides the SSRF protection. The theoretical DNS-rebinding window
    between that check and the actual connect() is negligible for this workload.
    """

    def __init__(self, validated_ips: list[str], hostname: str, **kwargs):
        self._validated_ips = validated_ips
        self._hostname = hostname
        super().__init__(**kwargs)

    # No send() override — let requests use normal DNS + TLS verification.


def _hard_block_check(url: str) -> tuple[str, list[str]]:
    """Run Hard Rules 1-3 on a URL.

    Returns (normalized_hostname, validated_ips) on success.
    Raises ValueError with structured reason code on failure.
    """
    parsed = urlparse(url)

    # Rule 1: HTTPS only
    if (parsed.scheme or "").lower() != "https":
        log.warning("SSRF block reason=hard_block_scheme url=%s", url[:80])
        raise ValueError(f"hard_block_scheme: URL scheme '{parsed.scheme}' not allowed")

    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError("hard_block_scheme: empty hostname")

    # Rule 2: No bare IP-literal hostnames
    try:
        ipaddress.ip_address(hostname)
        log.warning("SSRF block reason=hard_block_ip_literal host=%s url=%s", hostname, url[:80])
        raise ValueError(f"hard_block_ip_literal: bare IP address '{hostname}' not allowed")
    except ValueError as exc:
        if "hard_block_ip_literal" in str(exc):
            raise
        # Not an IP address — this is good, it's a hostname

    # IDNA normalization
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        pass

    # Rule 3: DNS resolution — reject if any record is private
    validated_ips = _resolve_all_ips(hostname)

    return hostname, validated_ips


# ---------------------------------------------------------------------------
# SSRF Host Lists — persistent allowed/rejected hostname tracking
# ---------------------------------------------------------------------------

_ssrf_host_lock = threading.Lock()
_ssrf_allowed_hosts: dict[str, dict] = {}
_ssrf_rejected_hosts: dict[str, dict] = {}
_ssrf_lists_loaded = False


def _load_host_lists(cfg: dict) -> None:
    """Load allowed/rejected host lists from disk into memory."""
    global _ssrf_allowed_hosts, _ssrf_rejected_hosts, _ssrf_lists_loaded
    projects_root = Path(cfg.get("projects_root", "/data/shared"))

    for name, target in [
        ("_ssrf_allowed_hosts.json", "_allowed"),
        ("_ssrf_rejected_hosts.json", "_rejected"),
    ]:
        path = projects_root / name
        data: dict = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                data = raw.get("hosts", {})
                log.info("Loaded %d entries from %s", len(data), path)
            except Exception as exc:
                log.warning("Could not load %s: %s", path, exc)

        if target == "_allowed":
            _ssrf_allowed_hosts = data
        else:
            _ssrf_rejected_hosts = data

    _ssrf_lists_loaded = True

    # Startup warnings
    if len(_ssrf_allowed_hosts) > 50:
        log.warning("SSRF allowed_hosts has %d entries (>50) — consider manual audit",
                    len(_ssrf_allowed_hosts))
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for host, meta in _ssrf_allowed_hosts.items():
        added = meta.get("added_ts", "")
        expires = meta.get("expires_ts")
        if expires:
            try:
                if datetime.fromisoformat(expires) < now:
                    log.warning("SSRF allowed host %s has EXPIRED (expires_ts=%s)", host, expires)
            except Exception:
                pass
        elif added:
            try:
                age = (now - datetime.fromisoformat(added)).days
                if age > 180:
                    log.warning("SSRF allowed host %s is %d days old without expires_ts", host, age)
            except Exception:
                pass


def _save_host_list(cfg: dict, list_name: str, data: dict) -> None:
    """Atomically write a host list to disk."""
    import os as _os
    projects_root = Path(cfg.get("projects_root", "/data/shared"))
    path = projects_root / list_name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"hosts": data}, indent=2, ensure_ascii=False), encoding="utf-8")
    _os.replace(tmp, path)


def _add_allowed_host(hostname: str, meta: dict, cfg: dict) -> None:
    """Add a hostname to the allowed list and persist."""
    with _ssrf_host_lock:
        _ssrf_allowed_hosts[hostname] = meta
        _save_host_list(cfg, "_ssrf_allowed_hosts.json", _ssrf_allowed_hosts)
    log.info("SSRF host ALLOWED: %s  source=%s", hostname, meta.get("source", ""))


def _add_rejected_host(hostname: str, cfg: dict) -> None:
    """Add a hostname to the rejected list and persist."""
    from datetime import datetime, timezone
    with _ssrf_host_lock:
        _ssrf_rejected_hosts[hostname] = {
            "added_ts": datetime.now(timezone.utc).isoformat(),
            "added_by": "user",
        }
        _save_host_list(cfg, "_ssrf_rejected_hosts.json", _ssrf_rejected_hosts)
    log.info("SSRF host REJECTED: %s", hostname)


def _is_host_allowed(hostname: str) -> str:
    """Check hostname against static + dynamic allowlists and rejected list.

    Returns:
        "static"   — in DOWNLOAD_ALLOWLIST
        "dynamic"  — in _ssrf_allowed_hosts
        "rejected" — in _ssrf_rejected_hosts
        "unknown"  — not in any list
    """
    # Check rejected first
    if hostname in _ssrf_rejected_hosts:
        return "rejected"
    # Check static allowlist
    if any(fnmatch.fnmatch(hostname, p) for p in DOWNLOAD_ALLOWLIST):
        return "static"
    # Check dynamic allowlist (with expiry)
    if hostname in _ssrf_allowed_hosts:
        meta = _ssrf_allowed_hosts[hostname]
        expires = meta.get("expires_ts")
        if expires:
            from datetime import datetime, timezone
            try:
                if datetime.fromisoformat(expires) < datetime.now(timezone.utc):
                    return "unknown"  # expired — treat as unknown
            except Exception:
                pass
        return "dynamic"
    return "unknown"


# ---------------------------------------------------------------------------
# CC BY attribution helper
# ---------------------------------------------------------------------------

def _ensure_cc_by_attribution(info: dict) -> None:
    """Ensure CC BY candidates have attribution_text. Construct fallback if missing."""
    lic = info.get("license_summary", "")
    if lic.startswith("CC BY") and not info.get("attribution_text"):
        title = info.get("title", "")
        author = info.get("photographer", "") or info.get("author", "")
        source = info.get("source_site", "")
        info["attribution_text"] = (
            f'"{title}" by {author} / {source} / {lic}' if author
            else f'"{title}" / {source} / {lic}'
        )
        info["attribution_required"] = True
        log.warning("CC BY candidate missing attribution_text, constructed fallback: source=%s title=%s",
                    source, title[:60])


# ---------------------------------------------------------------------------
# License normalizers
# ---------------------------------------------------------------------------

def _normalize_collections(raw) -> list:
    if isinstance(raw, list):
        return [c.lower().strip() for c in raw]
    if raw:
        return [str(raw).lower().strip()]
    return []


def _normalize_license(licenseurl: str, collections: list) -> str:
    """Normalize an Internet Archive licenseurl + collections list to a short license string."""
    CC_MAP = {
        "creativecommons.org/publicdomain/zero":  "CC0",
        "creativecommons.org/licenses/by/":        "CC BY",
        "creativecommons.org/licenses/by-sa/":     "",
        "creativecommons.org/licenses/by-nd/":     "",
        "creativecommons.org/licenses/by-nc/":     "",
        "creativecommons.org/licenses/by-nc-sa/":  "",
        "creativecommons.org/licenses/by-nc-nd/":  "",
        "creativecommons.org/publicdomain/mark/":  "Public Domain",
    }
    if licenseurl:
        lu = licenseurl.lower()
        for pattern, short in CC_MAP.items():
            if pattern in lu:
                if not short:
                    return ""
                if short in ("CC0", "Public Domain"):
                    return short
                version = re.search(r"/(\d+\.\d+)/?", licenseurl)
                return f"{short} {version.group(1)}" if version else short
    if any(c in ALLOWLIST_COLLECTIONS for c in collections):
        return "Public Domain"
    return ""


def _normalize_openverse_license(result: dict) -> str:
    lic = (result.get("license") or "").lower()
    version = result.get("license_version") or ""
    LICENSE_DISPLAY = {
        "cc0": "CC0", "pdm": "Public Domain",
        "by": "CC BY", "by-sa": "CC BY-SA", "by-nd": "CC BY-ND",
        "by-nc": "CC BY-NC", "by-nc-sa": "CC BY-NC-SA", "by-nc-nd": "CC BY-NC-ND",
    }
    short = LICENSE_DISPLAY.get(lic, lic.upper())
    if version and short not in {"CC0", "Public Domain"}:
        return f"{short} {version}"
    return short


def _normalize_europeana_license(rights_uri: str) -> str:
    uri = (rights_uri or "").lower()
    if "publicdomain/zero" in uri:    return "CC0"
    if "publicdomain/mark" in uri:    return "Public Domain"
    # CC BY (attribution only) — commercially usable
    if "/licenses/by/4" in uri:       return "CC BY 4.0"
    if "/licenses/by/3" in uri:       return "CC BY 3.0"
    if "/licenses/by/2" in uri:       return "CC BY 2.0"
    if "/licenses/by/1" in uri:       return "CC BY 1.0"
    # CC BY-SA (attribution + ShareAlike) — commercially usable; dominates Wikimedia/Europeana
    # Must be matched BEFORE /by-nd/ and /by-nc/ to avoid false prefix match.
    if "/licenses/by-sa/" in uri:
        m = re.search(r"/by-sa/(\d+)", uri)
        ver = m.group(1) if m else "4"
        return f"CC BY-SA {ver}.0"
    # Non-commercial or NoDerivatives — not accepted for commercial use
    if "/licenses/by-nd/" in uri:     return ""
    if "/licenses/by-nc" in uri:      return ""
    if "rightsstatements.org" in uri:
        parts = [p for p in uri.rstrip("/").split("/") if p]
        slug = parts[-2] if len(parts) >= 2 else parts[-1] if parts else ""
        if slug.upper() == "PDM":
            return "Public Domain"
        return ""
    return ""


def _normalize_wikimedia_license(short_name: str) -> str:
    s = short_name.strip()
    if s.lower() in {"public domain", "pd"}: return "Public Domain"
    if s.lower() in {"cc0", "cc 0"}:         return "CC0"
    if s.lower().startswith("pd-"):
        return "Public Domain"
    return s


def _normalize_freesound_license(lic: str) -> str:
    if not lic:
        return ""
    lic_lower = lic.lower()
    # URL format — what the Freesound API actually returns in sound objects
    # e.g. "https://creativecommons.org/publicdomain/zero/1.0/"
    #      "https://creativecommons.org/licenses/by/4.0/"
    if "publicdomain/zero" in lic_lower:
        return "CC0"
    if "licenses/by/" in lic_lower:          # matches /by/ but NOT /by-sa/, /by-nc/, etc.
        version = re.search(r"/(\d+\.\d+)/?", lic)
        return f"CC BY {version.group(1)}" if version else "CC BY"
    # Legacy display-name format (kept for safety)
    if lic == "Creative Commons 0":        return "CC0"
    if lic == "Attribution":               return "CC BY"
    if lic == "Attribution NonCommercial": return ""
    return ""


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _normalize_creator(creator) -> str:
    if isinstance(creator, list): return ", ".join(str(c) for c in creator)
    return str(creator) if creator else ""


def _normalize_subject(subject) -> list:
    if isinstance(subject, list): return [str(s) for s in subject]
    s = str(subject) if subject else ""
    if ";" in s: return [x.strip() for x in s.split(";") if x.strip()]
    if "|" in s: return [x.strip() for x in s.split("|") if x.strip()]
    return [s] if s else []


def _normalize_url_for_dedup(url: str) -> str:
    if not url: return ""
    try:
        p = urlparse(url.lower())
        return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url.lower().rstrip("/")


# ---------------------------------------------------------------------------
# Openverse OAuth2 token management
# ---------------------------------------------------------------------------

_openverse_token: dict = {"token": "", "expires_at": 0.0}
_openverse_lock = threading.Lock()
# Global semaphore: caps total concurrent Openverse HTTP requests across ALL
# parallel items.  Without this, 9 items × max_concurrent=4 = 36 simultaneous
# API calls which immediately triggers 429s.
_openverse_search_sem = threading.Semaphore(2)


def _get_openverse_token(api_keys: dict) -> str:
    """Get a valid Openverse Bearer token, refreshing proactively if near expiry."""
    client_id = api_keys.get("openverse_client_id", "")
    client_secret = api_keys.get("openverse_client_secret", "")
    if not client_id or not client_secret:
        return ""
    with _openverse_lock:
        now = time.monotonic()
        if _openverse_token["token"] and now < _openverse_token["expires_at"]:
            return _openverse_token["token"]
        try:
            r = requests.post(
                "https://api.openverse.org/v1/auth_tokens/token/",
                data={"client_id": client_id, "client_secret": client_secret,
                      "grant_type": "client_credentials"},
                headers={"User-Agent": _USER_AGENT},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            token = data["access_token"]
            expires_in = int(data.get("expires_in", 86400))
            _openverse_token["token"] = token
            _openverse_token["expires_at"] = now + expires_in - 60
            log.info("Openverse token refreshed; expires in %ds", expires_in)
            return token
        except Exception as exc:
            log.warning("Openverse token refresh failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# New source: Openverse images
# ---------------------------------------------------------------------------

def _source_search_openverse_images(
    api_keys: dict, query: str, per_page: int, backoff: list, extra_params: dict | None = None,
) -> list:
    key = ("openverse", "image", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache: return _search_cache[key]

    token = _get_openverse_token(api_keys)
    headers: dict = {"User-Agent": _USER_AGENT}
    if token: headers["Authorization"] = f"Bearer {token}"

    # Openverse page_size limits: authenticated = 50, unauthenticated = 20.
    # Sending page_size > 50 with a token returns 401 (not 400) — which is
    # the root cause of the 401 storm: all threads got 401 simultaneously,
    # triggering cascading token refreshes that 429'd the auth endpoint.
    max_page = 50 if token else 20
    # cc0, by only — CC BY-SA excluded (ShareAlike conflicts with YouTube ToS).
    params: dict = {"q": query, "license": "cc0,by",
                    "page_size": min(per_page, max_page), "filter_dead": "true", "extension": "jpg,png"}
    if extra_params: params.update(extra_params)

    def call():
        # Acquire global semaphore to cap cross-item concurrency at 2 total
        # (prevents 9 parallel items from firing 36 simultaneous API calls).
        with _openverse_search_sem:
            r = requests.get("https://api.openverse.org/v1/images/",
                             headers=headers, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json().get("results", [])

    log.info("Openverse search images q=%r per_page=%d", query, per_page)
    try:
        results = _with_backoff(call, backoff)
    except requests.HTTPError as exc:
        body = (exc.response.text or "")[:200] if exc.response is not None else ""
        log.warning("Openverse search failed q=%r: %s  body=%r", query, exc, body)
        return []  # do NOT cache failures — allow retry on next call
    except Exception as exc:
        log.warning("Openverse search failed q=%r: %s", query, exc)
        return []  # do NOT cache failures — allow retry on next call

    candidates = []
    for result in results:
        license_summary = _normalize_openverse_license(result)
        if not is_license_acceptable(license_summary): continue

        # Require a thumbnail URL from Openverse's own CDN.  Without it, the only
        # fallback is the raw provider URL (could be any host), which hits the SSRF
        # allowlist and returns "pending" for the vast majority of providers.
        thumbnail = result.get("thumbnail") or ""
        if not thumbnail:
            continue

        source_id = result.get("id", "")
        title = result.get("title") or ""
        author = result.get("creator") or ""
        attr_required = result.get("license") not in {"cc0", "pdm"}
        attr_text = result.get("attribution") or (
            f'"{title}" by {author} / openverse / {license_summary}' if author
            else f'"{title}" / openverse / {license_summary}')
        candidates.append({
            "source_site": "openverse", "source_id": source_id,
            "asset_page_url": result.get("foreign_landing_url", ""),
            "file_url": result.get("url", ""),
            "preview_url": thumbnail,
            "title": title, "description": "",
            "tags": [t["name"] for t in result.get("tags", []) if isinstance(t, dict)],
            "photographer": author, "license_summary": license_summary,
            "license_url": result.get("license_url", ""),
            "query_used": query,
            "width": result.get("width") or 0, "height": result.get("height") or 0,
            "attribution_required": attr_required, "attribution_text": attr_text,
            "provider": result.get("provider", ""),
        })
    _search_cache[key] = candidates
    return candidates


# ---------------------------------------------------------------------------
# New source: Wikimedia Commons images
# ---------------------------------------------------------------------------

def _source_search_wikimedia_images(
    api_keys: dict, query: str, per_page: int, backoff: list, extra_params: dict | None = None,
) -> list:
    key = ("wikimedia", "image", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache: return _search_cache[key]

    token = api_keys.get("wikimedia_api_token", "")
    headers: dict = {"User-Agent": _USER_AGENT}
    if token: headers["Authorization"] = f"Bearer {token}"

    params: dict = {
        "action": "query", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": query, "gsrlimit": min(per_page, 500),
        "prop": "imageinfo",
        "iiprop": "url|size|dimensions|extmetadata|mime|mediatype",
        "iiurlwidth": 800, "format": "json",
    }
    if extra_params: params.update(extra_params)

    def call():
        r = requests.get("https://commons.wikimedia.org/w/api.php",
                         headers=headers, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()

    log.info("Wikimedia search images q=%r per_page=%d", query, per_page)
    try: data = _with_backoff(call, backoff)
    except Exception as exc:
        log.warning("Wikimedia search failed: %s", exc)
        return []  # do NOT cache failures — allow retry on next call

    pages = (data.get("query") or {}).get("pages", {})
    ACCEPTED_MIMES = {"image/jpeg", "image/png", "image/gif"}
    candidates = []

    for page in pages.values():
        ii_list = page.get("imageinfo") or []
        if not ii_list: continue
        ii = ii_list[0]
        if ii.get("mime", "") not in ACCEPTED_MIMES: continue

        meta = ii.get("extmetadata", {})
        title = page.get("title", "").replace("File:", "", 1).strip()
        author = _strip_html((meta.get("Artist") or {}).get("value", ""))
        license_short_raw = (meta.get("LicenseShortName") or {}).get("value", "")
        license_summary = _normalize_wikimedia_license(license_short_raw)
        if not is_license_acceptable(license_summary): continue

        license_url = (meta.get("LicenseUrl") or {}).get("value", "")
        categories_raw = (meta.get("Categories") or {}).get("value", "")
        tags = [c.strip() for c in categories_raw.split("|") if c.strip()]
        asset_page = ii.get("descriptionurl", "")
        source_id = page.get("title", "")
        attr_required = (meta.get("AttributionRequired") or {}).get("value", "").lower() == "true"
        attr_text = (f'"{title}" by {author} / Wikimedia Commons / {license_summary}' if author
                     else f'"{title}" / Wikimedia Commons / {license_summary}')

        candidate = {
            "source_site": "wikimedia", "source_id": source_id,
            "asset_page_url": asset_page,
            "file_url": ii.get("url", ""),
            "preview_url": ii.get("thumburl") or ii.get("url", ""),
            "title": title, "description": "", "tags": tags,
            "photographer": author, "license_summary": license_summary,
            "license_url": license_url, "query_used": query,
            "width": ii.get("width", 0), "height": ii.get("height", 0),
            "attribution_required": attr_required, "attribution_text": attr_text,
        }
        if license_short_raw.lower().startswith("pd-art"):
            candidate["license_note"] = "PD-Art — jurisdiction-specific reproduction limits may apply"
        candidates.append(candidate)
    _search_cache[key] = candidates
    return candidates


# ---------------------------------------------------------------------------
# New source: Wikimedia Commons videos
# ---------------------------------------------------------------------------

def _source_search_wikimedia_videos(
    api_keys: dict, query: str, per_page: int, backoff: list, extra_params: dict | None = None,
) -> list:
    key = ("wikimedia", "video", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache: return _search_cache[key]

    token = api_keys.get("wikimedia_api_token", "")
    headers: dict = {"User-Agent": _USER_AGENT}
    if token: headers["Authorization"] = f"Bearer {token}"

    params: dict = {
        "action": "query", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"filetype:video {query}", "gsrlimit": min(per_page, 500),
        "prop": "imageinfo",
        "iiprop": "url|size|dimensions|extmetadata|mime|mediatype",
        "format": "json",
    }
    if extra_params: params.update(extra_params)

    def call():
        r = requests.get("https://commons.wikimedia.org/w/api.php",
                         headers=headers, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()

    log.info("Wikimedia search videos q=%r per_page=%d", query, per_page)
    try: data = _with_backoff(call, backoff)
    except Exception as exc:
        log.warning("Wikimedia video search failed: %s", exc)
        return []  # do NOT cache failures — allow retry on next call

    pages = (data.get("query") or {}).get("pages", {})
    ACCEPTED_MIMES = {"video/webm", "video/ogg", "video/mp4"}
    candidates = []

    for page in pages.values():
        ii_list = page.get("imageinfo") or []
        if not ii_list: continue
        ii = ii_list[0]
        mime = ii.get("mime", "")
        if mime not in ACCEPTED_MIMES: continue

        meta = ii.get("extmetadata", {})
        title = page.get("title", "").replace("File:", "", 1).strip()
        author = _strip_html((meta.get("Artist") or {}).get("value", ""))
        license_short_raw = (meta.get("LicenseShortName") or {}).get("value", "")
        license_summary = _normalize_wikimedia_license(license_short_raw)
        if not is_license_acceptable(license_summary): continue

        license_url = (meta.get("LicenseUrl") or {}).get("value", "")
        categories_raw = (meta.get("Categories") or {}).get("value", "")
        tags = [c.strip() for c in categories_raw.split("|") if c.strip()]
        asset_page = ii.get("descriptionurl", "")
        source_id = page.get("title", "")
        attr_required = (meta.get("AttributionRequired") or {}).get("value", "").lower() == "true"
        attr_text = (f'"{title}" by {author} / Wikimedia Commons / {license_summary}' if author
                     else f'"{title}" / Wikimedia Commons / {license_summary}')

        candidate = {
            "source_site": "wikimedia", "source_id": source_id,
            "asset_page_url": asset_page,
            "file_url": ii.get("url", ""),
            "preview_url": "",
            "title": title, "description": "", "tags": tags,
            "photographer": author, "license_summary": license_summary,
            "license_url": license_url, "query_used": query,
            "width": ii.get("width", 0), "height": ii.get("height", 0),
            "mime": mime,
            "attribution_required": attr_required, "attribution_text": attr_text,
        }
        if license_short_raw.lower().startswith("pd-art"):
            candidate["license_note"] = "PD-Art — jurisdiction-specific reproduction limits may apply"
        candidates.append(candidate)
    _search_cache[key] = candidates
    return candidates


# ---------------------------------------------------------------------------
# New source: Europeana images
# ---------------------------------------------------------------------------

def _source_search_europeana_images(
    api_keys: dict, query: str, per_page: int, backoff: list, extra_params: dict | None = None,
) -> list:
    europeana_key = api_keys.get("europeana", "")
    if not europeana_key:
        log.debug("Europeana API key not configured; skipping")
        return []

    key = ("europeana", "image", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache: return _search_cache[key]

    params: dict = {
        "wskey": europeana_key, "query": query, "qf": "TYPE:IMAGE",
        # "open" = CC0/PDM only.  "restricted" adds CC BY (and CC BY-SA/NC which
        # is_license_acceptable will then reject).  Using both broadens recall
        # significantly — many cultural heritage items carry CC BY, not CC0/PDM.
        "reusability": "open,restricted", "media": "true", "thumbnail": "true",
        "rows": min(per_page, 100), "cursor": "*", "profile": "rich",
    }
    if extra_params: params.update(extra_params)

    def call():
        r = requests.get("https://api.europeana.eu/record/v2/search.json",
                         headers={"User-Agent": _USER_AGENT}, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()

    log.info("Europeana search images q=%r per_page=%d", query, per_page)
    try: data = _with_backoff(call, backoff)
    except Exception as exc:
        log.warning("Europeana search failed: %s", exc)
        return []  # do NOT cache failures — allow retry on next call

    items = data.get("items") or []
    candidates = []
    for item in items:
        if item.get("previewNoDistribute"): continue
        url = (item.get("edmIsShownBy") or [""])[0]
        if not url: continue
        rights_uri = (item.get("rights") or [""])[0]
        license_summary = _normalize_europeana_license(rights_uri)
        if not is_license_acceptable(license_summary): continue

        preview_url = (item.get("edmPreview") or [""])[0]
        # Require Europeana's own CDN thumbnail.  The fallback (edmIsShownBy) is a
        # museum/institution server URL that is almost never in DOWNLOAD_ALLOWLIST.
        if not preview_url:
            continue
        title = (item.get("title") or [""])[0]
        author = ", ".join(item.get("dcCreator") or [])
        provider = (item.get("dataProvider") or [""])[0]
        guid = item.get("guid", "")
        source_id = item.get("id", "")
        attr_required = license_summary not in {"CC0", "Public Domain"}
        attr_text = (f'"{title}" by {author} / {provider} / {license_summary}' if author
                     else f'"{title}" / {provider} / {license_summary}')

        candidates.append({
            "source_site": "europeana", "source_id": source_id,
            "asset_page_url": guid,
            "file_url": url, "preview_url": preview_url,
            "title": title, "description": "", "tags": [],
            "photographer": author, "license_summary": license_summary,
            "license_url": rights_uri, "query_used": query,
            "width": 0, "height": 0,
            "attribution_required": attr_required, "attribution_text": attr_text,
            "provider": provider,
        })
    _search_cache[key] = candidates
    return candidates


# ---------------------------------------------------------------------------
# New source: Internet Archive images
# ---------------------------------------------------------------------------

def _source_search_archive_images(
    api_keys: dict, query: str, per_page: int, backoff: list, extra_params: dict | None = None,
) -> list:
    key = ("archive", "image", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache: return _search_cache[key]

    scrape_q = f"mediatype:image AND ({query}) AND licenseurl:(*creativecommons*)"
    params = {"q": scrape_q,
              "fields": "identifier,title,creator,subject,licenseurl,date,mediatype,collection",
              "count": max(100, min(per_page, 100))}  # Archive scrape API minimum is 100
    headers = {"User-Agent": _USER_AGENT}

    def call_scrape():
        r = requests.get("https://archive.org/services/search/v1/scrape",
                         headers=headers, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()

    log.info("Archive search images q=%r count=%d", query, per_page)
    try: scrape_data = _with_backoff(call_scrape, backoff)
    except Exception as exc:
        log.warning("Archive scrape failed: %s", exc)
        return []  # do NOT cache failures — allow retry on next call

    ACCEPTED_FORMATS = {"JPEG", "PNG", "GIF"}
    candidates = []

    for item in (scrape_data.get("items") or [])[:per_page]:
        identifier = item.get("identifier", "")
        if not identifier: continue
        try:
            meta_r = requests.get(f"https://archive.org/metadata/{identifier}",
                                   headers=headers, timeout=_TIMEOUT)
            meta_r.raise_for_status()
            meta_data = meta_r.json()
        except Exception as exc:
            log.debug("Archive metadata %s: %s", identifier, exc); continue

        metadata = meta_data.get("metadata", {})
        files = meta_data.get("files", [])
        cands_f = [f for f in files if f.get("format") in ACCEPTED_FORMATS]
        originals = [f for f in cands_f if f.get("source") == "original"]
        chosen = originals[0] if originals else (cands_f[0] if cands_f else None)
        if not chosen: continue

        license_raw = metadata.get("licenseurl", "") or metadata.get("license", "")
        collections = _normalize_collections(metadata.get("collection", []))
        license_summary = _normalize_license(license_raw, collections)
        if not is_license_acceptable(license_summary): continue

        title = str(metadata.get("title") or identifier)
        author = _normalize_creator(metadata.get("creator", ""))
        attr_required = license_summary not in {"CC0", "Public Domain"}
        attr_text = (f'"{title}" by {author} / archive.org / {license_summary}' if author
                     else f'"{title}" / archive.org / {license_summary}')

        candidates.append({
            "source_site": "archive", "source_id": identifier,
            "asset_page_url": f"https://archive.org/details/{identifier}",
            "file_url": f"https://archive.org/download/{identifier}/{chosen['name']}",
            "preview_url": f"https://archive.org/services/img/{identifier}",
            "title": title, "description": "",
            "tags": _normalize_subject(metadata.get("subject", "")),
            "photographer": author, "license_summary": license_summary,
            "license_url": license_raw, "query_used": query,
            "width": int(chosen.get("width", 0) or 0),
            "height": int(chosen.get("height", 0) or 0),
            "attribution_required": attr_required, "attribution_text": attr_text,
        })
    # Single politeness delay after fetching all metadata — not per-item.
    if candidates:
        time.sleep(random.uniform(0.2, 0.5))

    _search_cache[key] = candidates
    return candidates


# ---------------------------------------------------------------------------
# New source: Internet Archive videos
# ---------------------------------------------------------------------------

def _source_search_archive_videos(
    api_keys: dict, query: str, per_page: int, backoff: list, extra_params: dict | None = None,
) -> list:
    key = ("archive", "video", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache: return _search_cache[key]

    scrape_q = f"mediatype:movies AND ({query}) AND licenseurl:(*creativecommons*)"
    params = {"q": scrape_q,
              "fields": "identifier,title,creator,subject,licenseurl,date,mediatype,collection",
              "count": max(100, min(per_page, 100))}  # Archive scrape API minimum is 100
    headers = {"User-Agent": _USER_AGENT}

    def call_scrape():
        r = requests.get("https://archive.org/services/search/v1/scrape",
                         headers=headers, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()

    log.info("Archive search videos q=%r count=%d", query, per_page)
    try: scrape_data = _with_backoff(call_scrape, backoff)
    except Exception as exc:
        log.warning("Archive scrape videos failed: %s", exc)
        return []  # do NOT cache failures — allow retry on next call

    VIDEO_FMT_PRIORITY = ["h.264", "MPEG4", "512Kb MPEG4", "H.264 IA"]
    candidates = []

    for item in (scrape_data.get("items") or [])[:per_page]:
        identifier = item.get("identifier", "")
        if not identifier: continue
        try:
            meta_r = requests.get(f"https://archive.org/metadata/{identifier}",
                                   headers=headers, timeout=_TIMEOUT)
            meta_r.raise_for_status()
            meta_data = meta_r.json()
        except Exception as exc:
            log.debug("Archive video metadata %s: %s", identifier, exc); continue

        metadata = meta_data.get("metadata", {})
        files = meta_data.get("files", [])
        chosen = None
        for fmt in VIDEO_FMT_PRIORITY:
            found = [f for f in files if f.get("format") == fmt]
            if found: chosen = found[0]; break
        if not chosen:
            chosen = next((f for f in files if str(f.get("name","")).endswith(".mp4")), None)
        if not chosen: continue

        license_raw = metadata.get("licenseurl", "") or metadata.get("license", "")
        collections = _normalize_collections(metadata.get("collection", []))
        license_summary = _normalize_license(license_raw, collections)
        if not is_license_acceptable(license_summary): continue

        title = str(metadata.get("title") or identifier)
        author = _normalize_creator(metadata.get("creator", ""))
        attr_required = license_summary not in {"CC0", "Public Domain"}
        attr_text = (f'"{title}" by {author} / archive.org / {license_summary}' if author
                     else f'"{title}" / archive.org / {license_summary}')

        candidates.append({
            "source_site": "archive", "source_id": identifier,
            "asset_page_url": f"https://archive.org/details/{identifier}",
            "file_url": f"https://archive.org/download/{identifier}/{chosen['name']}",
            "preview_url": f"https://archive.org/services/img/{identifier}",
            "title": title, "description": "",
            "tags": _normalize_subject(metadata.get("subject", "")),
            "photographer": author, "license_summary": license_summary,
            "license_url": license_raw, "query_used": query,
            "width": int(chosen.get("width", 0) or 0),
            "height": int(chosen.get("height", 0) or 0),
            "duration_sec": float(chosen.get("length", 0) or 0),
            "attribution_required": attr_required, "attribution_text": attr_text,
        })
    # Single politeness delay after fetching all metadata — not per-item.
    if candidates:
        time.sleep(random.uniform(0.2, 0.5))

    _search_cache[key] = candidates
    return candidates


# ---------------------------------------------------------------------------
# SFX search — Freesound + Openverse Audio
# ---------------------------------------------------------------------------

def fetch_sfx(
    query: str,
    duration_sec: float,
    api_keys: dict,
    cfg: dict,
) -> list:
    """
    Search Freesound and Openverse Audio for sound effects.
    Returns deduplicated list of candidate dicts.
    """
    backoff = cfg.get("backoff_seconds", [5, 15, 45])
    sfx_limits = cfg.get("sfx_source_limits", {})
    max_dur = max(duration_sec * 2, 15)

    results: list = []

    # --- Freesound ---
    freesound_key = api_keys.get("freesound", "")
    fs_limit = (sfx_limits.get("freesound") or {}).get("candidates_sfx", 20)
    if freesound_key and fs_limit > 0:
        try:
            lic_filter = 'license:("Creative Commons 0" OR "Attribution")'
            # Freesound AND-matches every word; >4 words over-constrains and returns 0
            fs_query = " ".join(query.split()[:4])
            fs_params = {
                "token": freesound_key,
                "query": fs_query,
                "filter": lic_filter,
                "sort": "score",
                "fields": "id,name,duration,license,previews,tags,username,avg_rating,num_downloads,type,images,url,channels,loopable,single_event",
                "page_size": min(fs_limit, 150),
            }
            r = requests.get("https://freesound.org/apiv2/search/text/",
                             params=fs_params, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT)
            r.raise_for_status()
            for result in r.json().get("results", []):
                lic = _normalize_freesound_license(result.get("license", ""))
                if not is_license_acceptable(lic): continue
                sound_id = str(result.get("id", ""))
                username = result.get("username", "")
                name = result.get("name", "")
                asset_page = result.get("url") or f"https://freesound.org/people/{username}/sounds/{sound_id}/"
                previews = result.get("previews") or {}
                preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3", "")
                waveform_img = (result.get("images") or {}).get("waveform_m", "")
                results.append({
                    "source_site": "freesound", "source_id": sound_id,
                    "preview_url": preview_url,
                    "duration_sec": float(result.get("duration", 0)),
                    "title": name,
                    "tags": result.get("tags", []),
                    "license_summary": lic,
                    "license_url": f"https://creativecommons.org/licenses/{'zero/1.0' if lic == 'CC0' else 'by/4.0'}/",
                    "attribution_text": f'"{name}" by {username} / freesound.org / {lic}',
                    "author": username,
                    "rating": float(result.get("avg_rating", 0)),
                    "downloads": int(result.get("num_downloads", 0)),
                    "waveform_img": waveform_img,
                    "asset_page_url": asset_page,
                    "attribution_required": lic != "CC0",
                    "loopable": result.get("loopable", False),
                    "channels": result.get("channels", 2),
                })
        except Exception as exc:
            log.warning("Freesound search q=%r failed: %s", query, exc)

    # --- Openverse Audio ---
    ov_limit = (sfx_limits.get("openverse_audio") or {}).get("candidates_sfx", 10)
    if ov_limit > 0:
        try:
            token = _get_openverse_token(api_keys)
            if duration_sec < 30:    dur_bucket = "short"
            elif duration_sec < 120: dur_bucket = "medium"
            else:                    dur_bucket = "long"
            ov_headers: dict = {"User-Agent": _USER_AGENT}
            if token: ov_headers["Authorization"] = f"Bearer {token}"
            ov_max_page = 500 if token else 20  # anon limit is 20
            ov_params = {
                "q": query, "license": "cc0,by",
                "duration": dur_bucket, "source": "freesound,wikimedia_audio",
                "page_size": min(ov_limit, ov_max_page),
            }
            def _ov_audio_call():
                r = requests.get("https://api.openverse.org/v1/audio/",
                                 params=ov_params, headers=ov_headers, timeout=_TIMEOUT)
                r.raise_for_status()
                return r.json().get("results", [])

            for result in _with_backoff(_ov_audio_call, backoff):
                lic = _normalize_openverse_license(result)
                if not is_license_acceptable(lic): continue
                source_id = result.get("id", "")
                title = result.get("title") or ""
                author = result.get("creator") or ""
                preview_url = result.get("url", "")
                asset_page = result.get("foreign_landing_url", "")
                attribution_text = result.get("attribution") or (
                    f'"{title}" by {author} / openverse_audio / {lic}' if author
                    else f'"{title}" / openverse_audio / {lic}')
                results.append({
                    "source_site": "openverse_audio", "source_id": source_id,
                    "preview_url": preview_url,
                    "duration_sec": float(result.get("duration", 0)) / 1000.0,
                    "title": title,
                    "tags": [t["name"] for t in result.get("tags", []) if isinstance(t, dict)],
                    "license_summary": lic,
                    "license_url": result.get("license_url", ""),
                    "attribution_text": attribution_text,
                    "author": author, "rating": 0.0, "downloads": 0,
                    "waveform_img": result.get("waveform", ""),
                    "asset_page_url": asset_page,
                    "attribution_required": lic != "CC0",
                    "provider": result.get("provider", ""),
                })
        except Exception as exc:
            log.warning("Openverse audio search q=%r failed: %s", query, exc)

    # 3-pass dedup
    seen_source_ids: set = set()
    seen_landing_urls: set = set()
    seen_fallback: set = set()
    deduped = []
    for c in results:
        sid = (c["source_site"], c["source_id"])
        if c["source_id"] and sid in seen_source_ids: continue
        if c["source_id"]: seen_source_ids.add(sid)
        norm_page = _normalize_url_for_dedup(c.get("asset_page_url", ""))
        if norm_page and norm_page in seen_landing_urls: continue
        if norm_page: seen_landing_urls.add(norm_page)
        fb = (c["title"].lower()[:50], round(c.get("duration_sec", 0), 1))
        if not c["source_id"] and not norm_page:
            if fb in seen_fallback: continue
            seen_fallback.add(fb)
        deduped.append(c)
    return deduped


def _slug_to_title(page_url: str) -> str:
    """Extract human-readable title from Pexels/Pixabay page URL slug.
    e.g. https://www.pexels.com/photo/ancient-roman-temple-ruins-in-pompeii-italy-30204931/
         → "Ancient Roman Temple Ruins In Pompeii Italy"
    """
    slug = page_url.rstrip("/").split("/")[-1]
    parts = slug.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        slug = parts[0]
    return slug.replace("-", " ").title()


def _write_info_sidecar(path: Path, info: dict) -> None:
    """Write metadata sidecar alongside a downloaded file. Skips if already exists."""
    sidecar = Path(str(path) + ".info.json")
    if not sidecar.exists():
        try:
            sidecar.write_text(json.dumps(info, ensure_ascii=False))
        except Exception as exc:
            log.warning("Could not write info sidecar %s: %s", sidecar, exc)
_TIMEOUT    = 45  # seconds for API calls and downloads

# In-process search-response cache: (source, media_type, query, per_page) → list[dict]
_search_cache: dict[tuple, list] = {}


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, headers: Optional[dict] = None, validated_ips: list[str] | None = None,
         source: str = "", **kwargs) -> requests.Response:
    """HTTP GET with SSRF-safe redirect handling.

    If validated_ips is provided, uses _BoundHostAdapter to connect directly
    to pre-validated IPs (prevents DNS rebinding). Manual redirect loop
    re-validates each hop against hard rules + allowlist.
    """
    hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)

    # Use separate connect/read timeouts
    timeout = kwargs.pop("timeout", (10, 45))

    session = requests.Session()

    if validated_ips:
        hostname = urlparse(url).hostname or ""
        adapter = _BoundHostAdapter(validated_ips, hostname)
        session.mount("https://", adapter)

    # Manual redirect loop (Rule 4)
    kwargs["allow_redirects"] = False
    for hop in range(_MAX_REDIRECT_DEPTH + 1):
        resp = session.get(url, headers=hdrs, timeout=timeout, **kwargs)

        if resp.status_code not in (301, 302, 307, 308):
            return resp

        location = resp.headers.get("Location", "")
        if not location:
            return resp

        # Re-validate the redirect target
        try:
            new_hostname, new_ips = _hard_block_check(location)
        except ValueError as exc:
            log.warning("SSRF block reason=hard_block_redirect url=%s location=%s source=%s: %s",
                        url[:80], location[:80], source, exc)
            raise ValueError(f"hard_block_redirect: redirect to {location[:80]} blocked") from exc

        # Check allowlist for redirect target
        status = _is_host_allowed(new_hostname)
        if status == "rejected":
            log.warning("SSRF block reason=rejected_host host=%s url=%s source=%s",
                        new_hostname, location[:80], source)
            raise ValueError(f"rejected_host: redirect target {new_hostname} is rejected")
        if status == "unknown":
            log.warning("SSRF block reason=pending_host_review host=%s url=%s source=%s",
                        new_hostname, location[:80], source)
            raise ValueError(f"pending_host_review: redirect target {new_hostname} not in allowlist")

        # Update for next hop
        url = location
        validated_ips = new_ips
        if validated_ips:
            hostname = new_hostname
            adapter = _BoundHostAdapter(validated_ips, hostname)
            session = requests.Session()
            session.mount("https://", adapter)

    raise ValueError(f"hard_block_redirect: exceeded {_MAX_REDIRECT_DEPTH} redirects")


def _download_file(url: str, dest: Path, headers: Optional[dict] = None,
                   cfg: dict | None = None, source: str = "",
                   media_type: str = "image") -> str | None:
    """Stream-download url → dest with full SSRF protection.

    Returns None on success, "pending" if hostname is unknown and needs review.
    Raises ValueError on hard blocks or validation failures.
    """
    cfg = cfg or {}
    hostname = urlparse(url).hostname or ""

    # Step 1-2: Hard block checks (scheme, IP literal, DNS)
    try:
        norm_hostname, validated_ips = _hard_block_check(url)
    except ValueError:
        raise  # already logged by _hard_block_check

    # Step 3: Check rejected hosts
    host_status = _is_host_allowed(norm_hostname)
    if host_status == "rejected":
        log.warning("SSRF block reason=rejected_host host=%s url=%s source=%s",
                    norm_hostname, url[:80], source)
        raise ValueError(f"rejected_host: {norm_hostname}")

    # Step 4-5: Check static + dynamic allowlists
    if host_status == "unknown":
        log.warning("SSRF block reason=pending_host_review host=%s url=%s source=%s",
                    norm_hostname, url[:80], source)
        return "pending"

    # Host is allowed (static or dynamic) — proceed with download
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Determine size limit based on media type
    if media_type == "video":
        max_bytes = cfg.get("max_download_bytes_video", 2 * 1024 * 1024 * 1024)
    elif media_type == "audio":
        max_bytes = cfg.get("sfx_max_download_bytes", 20 * 1024 * 1024)
    else:
        max_bytes = cfg.get("max_download_bytes_image", 50 * 1024 * 1024)

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        r = _get(url, headers=headers, validated_ips=validated_ips, source=source, stream=True)
        r.raise_for_status()

        # Rule 5a: Pre-check Content-Length
        cl = r.headers.get("Content-Length")
        if cl:
            try:
                if int(cl) > max_bytes:
                    log.warning("SSRF block reason=file_size_exceeded host=%s url=%s cl=%s max=%d source=%s",
                                norm_hostname, url[:80], cl, max_bytes, source)
                    raise ValueError(f"file_size_exceeded: Content-Length {cl} > {max_bytes}")
            except ValueError as exc:
                if "file_size_exceeded" in str(exc):
                    raise

        # Content-Type pre-check
        ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ct:
            if ct in _BLOCKED_CONTENT_TYPES:
                log.warning("SSRF block reason=invalid_content_type host=%s url=%s ct=%s source=%s",
                            norm_hostname, url[:80], ct, source)
                raise ValueError(f"invalid_content_type: {ct}")
            # Allow: image/*, audio/*, video/*, application/octet-stream, or absent
            if (not ct.startswith(("image/", "audio/", "video/"))
                    and ct != "application/octet-stream"):
                log.warning("SSRF block reason=invalid_content_type host=%s url=%s ct=%s source=%s",
                            norm_hostname, url[:80], ct, source)
                raise ValueError(f"invalid_content_type: {ct}")

        # Stream download with size enforcement (Rule 5b)
        total_bytes = 0
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if chunk:
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        log.warning("SSRF block reason=file_size_exceeded host=%s url=%s streamed=%d max=%d source=%s",
                                    norm_hostname, url[:80], total_bytes, max_bytes, source)
                        raise ValueError(f"file_size_exceeded: streamed {total_bytes} > {max_bytes}")
                    fh.write(chunk)

        # Magic-bytes post-check
        _check_magic_bytes(tmp, media_type)

        # Success — atomic rename
        tmp.replace(dest)
        log.debug("Downloaded %s → %s (%d bytes)", url[:80], dest.name, dest.stat().st_size)
        return None

    except Exception:
        # Clean up .part file on ANY failure
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# Magic byte signatures for file validation
_MAGIC_IMAGE = {
    "jpeg": b"\xff\xd8\xff",
    "png":  b"\x89PNG\r\n\x1a\n",
    "gif":  b"GIF8",
}
_MAGIC_WEBP_RIFF = b"RIFF"
_MAGIC_WEBP_WEBP = b"WEBP"

_MAGIC_AUDIO = {
    "id3":  b"ID3",
    "ogg":  b"OggS",
    "flac": b"fLaC",
}
_MAGIC_AUDIO_MPEG_SYNC = (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
_MAGIC_AUDIO_WAV_RIFF = b"RIFF"
_MAGIC_AUDIO_WAV_WAVE = b"WAVE"

_MAGIC_VIDEO_FTYP = b"ftyp"
_MAGIC_VIDEO_WEBM = b"\x1aE\xdf\xa3"


def _check_magic_bytes(path: Path, media_type: str) -> None:
    """Validate file header bytes match expected media type.

    Raises ValueError (reason=invalid_signature) on mismatch.
    """
    try:
        with path.open("rb") as f:
            header = f.read(12)
    except OSError as exc:
        raise ValueError(f"invalid_signature: cannot read file header: {exc}") from exc

    if len(header) == 0:
        raise ValueError("invalid_signature: empty file")

    if media_type == "image":
        # JPEG
        if header[:3] == _MAGIC_IMAGE["jpeg"]:
            return
        # PNG
        if header[:8] == _MAGIC_IMAGE["png"]:
            return
        # GIF
        if header[:4] == _MAGIC_IMAGE["gif"]:
            return
        # WebP: RIFF....WEBP
        if header[:4] == _MAGIC_WEBP_RIFF and header[8:12] == _MAGIC_WEBP_WEBP:
            return
        log.warning("SSRF block reason=invalid_signature path=%s media_type=%s header=%s",
                    path.name, media_type, header[:12].hex())
        raise ValueError(f"invalid_signature: image file header does not match any known format")

    elif media_type == "audio":
        # MP3 sync word
        if header[:2] in _MAGIC_AUDIO_MPEG_SYNC:
            return
        # ID3 tag (MP3 with metadata)
        if header[:3] == _MAGIC_AUDIO["id3"]:
            return
        # OGG
        if header[:4] == _MAGIC_AUDIO["ogg"]:
            return
        # FLAC
        if header[:4] == _MAGIC_AUDIO["flac"]:
            return
        # WAV: RIFF....WAVE
        if header[:4] == _MAGIC_AUDIO_WAV_RIFF and header[8:12] == _MAGIC_AUDIO_WAV_WAVE:
            return
        log.warning("SSRF block reason=invalid_signature path=%s media_type=%s header=%s",
                    path.name, media_type, header[:12].hex())
        raise ValueError(f"invalid_signature: audio file header does not match any known format")

    elif media_type == "video":
        # Try ffprobe first (already a project dependency)
        try:
            probe_result = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-show_entries", "format=format_name:stream=codec_type,duration",
                 "-of", "json", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if probe_result.returncode == 0:
                import json as _json
                probe_data = _json.loads(probe_result.stdout)
                fmt = (probe_data.get("format") or {}).get("format_name", "")
                allowed_fmts = {"mp4", "mov", "webm", "avi", "matroska",
                                "mov,mp4,m4a,3gp,3g2,mj2"}  # ffprobe compound name
                if not any(f in fmt for f in {"mp4", "mov", "webm", "avi", "matroska"}):
                    raise ValueError(f"invalid_signature: video format '{fmt}' not allowed")
                streams = probe_data.get("streams") or []
                has_video = any(s.get("codec_type") == "video" for s in streams)
                if not has_video:
                    raise ValueError("invalid_signature: no video stream found")
                return
        except FileNotFoundError:
            pass  # ffprobe not available — fall back to magic bytes
        except subprocess.TimeoutExpired:
            pass  # ffprobe hung — fall back
        except ValueError:
            raise  # re-raise our own validation errors
        except Exception:
            pass  # any other ffprobe error — fall back

        # Fallback: check magic bytes
        # MP4/MOV: ftyp box at offset 4
        if header[4:8] == _MAGIC_VIDEO_FTYP:
            return
        # WebM/MKV: EBML header
        if header[:4] == _MAGIC_VIDEO_WEBM:
            return
        log.warning("SSRF block reason=invalid_signature path=%s media_type=%s header=%s",
                    path.name, media_type, header[:12].hex())
        raise ValueError(f"invalid_signature: video file header does not match any known format")


def _with_backoff(fn, backoff_seconds: list[int]):
    """
    Call fn().  On HTTP 429, retry after each wait in backoff_seconds.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for wait in [0] + list(backoff_seconds):
        if wait:
            log.warning("429 received — retrying in %ds …", wait)
            time.sleep(wait)
        try:
            return fn()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                last_exc = exc
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _jitter(cfg: dict, source: str) -> None:
    """Sleep a random delay in [delay_ms_min, delay_ms_max] for the given source."""
    rl = cfg.get("rate_limits", {}).get(source, {})
    lo = float(rl.get("delay_ms_min", 0))
    hi = float(rl.get("delay_ms_max", 0))
    if hi > 0:
        time.sleep(random.uniform(lo, hi) / 1000.0)


def _run_download_tasks(
    tasks:       list[dict],
    cfg:         dict,
    max_workers: int | None = None,
    label:       str        = "",
) -> tuple[list[tuple[Path, dict]], dict[str, str], list[dict]]:
    """
    Download a batch of files in parallel using ThreadPoolExecutor.

    Each task dict must contain:
        dest    : Path  — local destination path
        url     : str   — URL to download
        info    : dict  — metadata written as .info.json sidecar
    Optional keys:
        headers : dict  — extra HTTP headers (e.g. Authorization for Pexels)
        thumb   : str   — thumbnail URL recorded for scorer pass-1.5

    Returns:
        saved              : list[(Path, info)] for each successfully saved file
        thumbs             : {str(dest): thumb_url} for thumbnail registration
        pending_candidates : list[dict] — candidates whose hostname needs review
    """
    tag = f"[{label}]" if label else "[download]"

    saved:              list[tuple[Path, dict]] = []
    thumbs:             dict[str, str]          = {}
    pending_candidates: list[dict]              = []
    pending:            list[dict]              = []

    # Fast path: files that already exist skip the download entirely.
    for task in tasks:
        dest = task["dest"]
        if dest.exists():
            _write_info_sidecar(dest, task["info"])
            saved.append((dest, task["info"]))
            if task.get("thumb"):
                thumbs[str(dest)] = task["thumb"]
        else:
            pending.append(task)

    cached = len(tasks) - len(pending)
    if not pending:
        log.info("  %s  total=%d  cached=%d  pending=0  (all cached)", tag, len(tasks), cached)
        return saved, thumbs, pending_candidates

    # Determine source and media_type from label (e.g. "img/pexels", "vid/wikimedia")
    _label_parts = label.split("/")
    _dl_media_type = "video" if _label_parts[0] == "vid" else "image"
    _dl_source = _label_parts[1] if len(_label_parts) > 1 else ""

    # Per-source download concurrency cap (e.g. Wikimedia throttles bulk downloads).
    # Read from rate_limits.<source>.max_download_concurrent; fall back to global download_workers.
    _src_dl_limit = (
        cfg.get("rate_limits", {}).get(_dl_source, {}).get("max_download_concurrent")
        if _dl_source else None
    )
    n_workers = min(max_workers or _src_dl_limit or cfg.get("download_workers", 8), len(pending))
    log.info("  %s  total=%d  cached=%d  pending=%d  workers=%d",
             tag, len(tasks), cached, len(pending), n_workers)
    t0 = time.perf_counter()

    # CC BY attribution check for all tasks before downloading
    for task in pending:
        _ensure_cc_by_attribution(task["info"])

    def _do(task: dict) -> tuple[dict, str]:
        """Returns (task, status) where status is 'ok', 'pending', or 'failed'.

        Per-host global semaphore (e.g. upload.wikimedia.org) limits concurrent
        connections across ALL download threads in the process.  This prevents
        Wikimedia's CDN rate-limit (429) without slowing down Pexels/Pixabay.

        On HTTP 429 the download is retried up to 3 times with increasing delays
        (3 s, 8 s, 20 s) for genuine rate-limits.  Robot-policy violations are
        not retried — they fail immediately.
        """
        dl_hostname = urlparse(task["url"]).hostname or ""
        limiter     = _get_host_limiter(dl_hostname, cfg)

        # Exponential backoff delays on 429: 2s, 4s, 8s, 16s
        _429_waits = (2, 4, 8, 16)
        for _attempt, _wait in enumerate((*_429_waits, None)):
            if limiter:
                limiter.acquire()
            try:
                result = _download_file(
                    task["url"], task["dest"],
                    headers=task.get("headers") or {},
                    cfg=cfg, source=_dl_source,
                    media_type=_dl_media_type,
                )
                if result == "pending":
                    return task, "pending"
                return task, "ok"
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    body = exc.response.text or ""
                    # Wikimedia "robot policy violation" — permanent UA-level block,
                    # retrying never succeeds.  Fail immediately.
                    is_robot_policy = "robot policy" in body.lower() or "0fa5166" in body
                    if not is_robot_policy and _wait is not None:
                        log.debug("  %s  429 rate-limit on %s — retry %d/%d in %ds",
                                  tag, task["dest"].name, _attempt + 1, len(_429_waits), _wait)
                        time.sleep(_wait)
                        continue
                    if is_robot_policy:
                        log.warning("  %s  skipped %s: Wikimedia robot policy block (UA rejected)",
                                    tag, task["dest"].name)
                log.warning("  %s  failed %s: %s", tag, task["dest"].name, exc)
                return task, "failed"
            except Exception as exc:  # noqa: BLE001
                log.warning("  %s  failed %s: %s", tag, task["dest"].name, exc)
                return task, "failed"
            finally:
                if limiter:
                    limiter.release()
        return task, "failed"  # exhausted retries

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for task, status in ex.map(_do, pending):
            if status == "ok":
                dest = task["dest"]
                _write_info_sidecar(dest, task["info"])
                saved.append((dest, task["info"]))
                if task.get("thumb"):
                    thumbs[str(dest)] = task["thumb"]
            elif status == "pending":
                # Build pending candidate entry from task info
                info = task["info"]
                hostname = urlparse(task["url"]).hostname or ""
                pending_candidates.append({
                    "hostname":        hostname,
                    "url":             task["url"],
                    "asset_id":        info.get("source_id", ""),
                    "source":          info.get("source_site", _dl_source),
                    "source_id":       info.get("source_id", ""),
                    "license_summary": info.get("license_summary", ""),
                    "title":           info.get("title", ""),
                    "preview_url":     info.get("preview_url", ""),
                    "info":            info,
                })

    n_pending = len(pending_candidates)
    log.info("  %s  saved=%d/%d  pending_review=%d  elapsed=%.1fs",
             tag, len(saved) - cached, len(pending), n_pending, time.perf_counter() - t0)
    return saved, thumbs, pending_candidates


# ---------------------------------------------------------------------------
# Query rotation helper
# ---------------------------------------------------------------------------

def _get_queries(
    search_prompt:   str,
    item:            dict | None,
    n:               int,
    inject_location: bool = True,
) -> list[tuple[str, int]]:
    """
    Returns a list of (query, per_query_n) tuples.

    B3: Location-specific include_keywords (capitalised words longer than 3 chars)
        are injected as a prefix into each query where the term is not already present.
        Only applied when inject_location=True (images only).
        Disabled for video queries — stock video sites have almost no location-tagged
        footage for rare historical subjects; injecting narrows the pool to 0.

    C2: First query gets 50% of budget; remaining queries share the rest equally.
        The first query in search_queries is the most specific by convention.

    Falls back to a single (search_prompt, n) tuple if no search_queries present.
    """
    if item:
        queries = list(item.get("search_queries") or [])
        if queries:
            # B3: inject location prefix for images only
            if inject_location:
                kws = item.get("include_keywords") or []
                location_terms = [kw for kw in kws
                                  if kw and kw[0].isupper() and len(kw) > 3]
                prefix = location_terms[0] if location_terms else ""
                if prefix:
                    queries = [
                        q if prefix.lower() in q.lower() else f"{prefix} {q}"
                        for q in queries
                    ]

            # C2: first query gets 50%, remaining queries share the rest
            first_n = max(1, n // 2)
            if len(queries) > 1:
                rest_n = max(1, math.ceil((n - first_n) / (len(queries) - 1)))
            else:
                first_n = n
                rest_n  = 0

            result = [(queries[0], first_n)]
            result += [(q, rest_n) for q in queries[1:]]
            return result

        # Fallback: apply location prefix to search_prompt for images only
        if inject_location:
            kws = item.get("include_keywords") or []
            loc = next((kw for kw in kws if kw and kw[0].isupper() and len(kw) > 3), "")
            if loc and loc.lower() not in search_prompt.lower():
                search_prompt = f"{loc} {search_prompt}"

    return [(search_prompt, n)]


# ---------------------------------------------------------------------------
# Query helpers for keyword-indexed sources
# ---------------------------------------------------------------------------

# Sources that use keyword/full-text indexing rather than semantic search.
# Long descriptive phrases (e.g. "Soviet reactor building dark dramatic night")
# match nothing in these databases.  _keyword_fallbacks() produces progressively
# shorter forms to retry when the full query returns 0 candidates.
_KEYWORD_INDEXED_SOURCES = frozenset({"openverse", "wikimedia", "europeana"})


def _keyword_fallbacks(query: str, item: dict | None = None) -> list[str]:
    """
    Return up to two shorter fallback queries for keyword-indexed databases.

    Strategy (each only added if different from the original and non-empty):
      1. First 4 words  — preserves the most specific phrase chunk
      2. First 2 words  — broadest catch-all
      3. item.include_keywords location term — direct keyword hit (e.g. "Chernobyl")
    """
    words = query.split()
    seen: set[str] = {query.lower()}
    result: list[str] = []

    for n in (4, 2):
        if len(words) > n:
            short = " ".join(words[:n])
            if short.lower() not in seen:
                seen.add(short.lower())
                result.append(short)

    if item:
        kws = item.get("include_keywords") or []
        loc = next((kw for kw in kws if kw and kw[0].isupper() and len(kw) > 3), "")
        if loc and loc.lower() not in seen:
            result.append(loc)

    return result


# ---------------------------------------------------------------------------
# Per-source query strategy
# ---------------------------------------------------------------------------

def _get_queries_for_source(
    search_prompt: str,
    item:          dict | None,
    n:             int,
    source:        str,
) -> list[tuple[str, int]]:
    """
    Return an optimised (query, per_query_n) list for keyword-indexed sources.

    Pexels and Pixabay use _get_queries() unchanged — their semantic search
    handles long descriptive phrases well.

    Openverse, Wikimedia, and Europeana are keyword / full-text indexed.
    Long LLM-generated phrases (e.g. "Soviet control room crisis alarm atmosphere")
    match nothing in those databases.  This function builds shorter, targeted
    queries for each source instead.

    Each query gets the full n budget so the deduplication pass (Phase 2) decides
    the final mix — rather than artificially capping per-query results and coming
    up short when some queries return fewer hits than their slice.

    Strategies
    ----------
    openverse  : 4-word truncated phrases (good keyword coverage on Flickr-heavy
                 content), location anchor appended as guaranteed top-up.
    wikimedia  : location anchor first (highest Public Domain hit rate; Commons is
                 title-indexed so one-word specifics outperform long phrases),
                 then 2-word truncated phrases for variety.
    europeana  : location anchor first (broadest CC BY recall), then 4-word phrases
                 (full-text metadata indexing handles slightly longer terms well).
    """
    raw_queries: list[str] = list((item or {}).get("search_queries") or []) or [search_prompt]

    # Location anchors: ALL capitalised include_keywords longer than 3 chars
    # (e.g. ["Chernobyl", "Pripyat"]).  Using all of them as separate queries
    # is important for Europeana where each proper noun has its own indexed cluster.
    loc = ""        # primary (first) — used by openverse/wikimedia
    all_locs: list[str] = []
    if item:
        kws = item.get("include_keywords") or []
        all_locs = [kw for kw in kws if kw and kw[0].isupper() and len(kw) > 3]
        loc = all_locs[0] if all_locs else ""

    def _trunc(q: str, words: int) -> str:
        return " ".join(q.split()[:words])

    def _dedup_build(phrases: list[str]) -> list[tuple[str, int]]:
        """Deduplicate while preserving order; assign each query the full n budget."""
        seen: set[str] = set()
        result: list[tuple[str, int]] = []
        for q in phrases:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                result.append((q, n))
        return result or [(search_prompt, n)]

    if source == "openverse":
        # Cap at 2 queries per item: 1 primary 4-word phrase + 1 location anchor.
        # page_size=50 already returns up to 50 candidates per query, which is
        # more than enough to fill the 20–30 image budget.  More queries add
        # API load with diminishing returns on result quality.
        primary = _trunc(raw_queries[0], 4) if raw_queries else search_prompt
        phrases = [primary]
        if loc and loc.lower() != primary.lower():
            phrases.append(loc)
        return _dedup_build(phrases)

    if source == "wikimedia":
        # Location anchor first (title-indexed → highest PD hit rate),
        # then 2-word phrases for variety.
        phrases = ([loc] if loc else []) + [_trunc(q, 2) for q in raw_queries]
        return _dedup_build(phrases)

    if source == "europeana":
        # ALL location keywords first (each proper noun is its own indexed cluster).
        # Then 2-word truncations (Europeana full-text search; shorter phrases recall more).
        # Then 4-word truncations as additional coverage.
        # This strategy is important: "Chernobyl" alone returns 17 images; adding
        # "Pripyat" and "exclusion zone" pushes past 20.
        phrases = (
            all_locs
            + [_trunc(q, 2) for q in raw_queries]
            + [_trunc(q, 4) for q in raw_queries]
        )
        return _dedup_build(phrases)

    # Should not reach here — only called for _KEYWORD_INDEXED_SOURCES.
    return [(search_prompt, n)]


# ---------------------------------------------------------------------------
# Source-filter helper
# ---------------------------------------------------------------------------

def _get_source_filters(item: dict | None, source: str) -> dict:
    """
    Return the source-specific filter overrides from item["source_filters"].
    Returns an empty dict if no overrides are defined for this source.
    """
    if item:
        sf = item.get("source_filters") or {}
        return dict(sf.get(source) or {})
    return {}


# ---------------------------------------------------------------------------
# Pexels — images
# ---------------------------------------------------------------------------

def _pexels_search_images(
    api_key: str,
    query: str,
    per_page: int,
    backoff: list[int],
    extra_params: dict | None = None,
) -> list[dict]:
    # Build a stable cache key that includes any extra params
    key = ("pexels", "image", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache:
        return _search_cache[key]

    headers = {"Authorization": api_key}
    q = " ".join(query.splitlines())

    # Always-on Pexels image defaults
    params: dict = {
        "query":       q,
        "per_page":    per_page,
        "orientation": "landscape",
        "size":        "large",
    }
    # Merge caller-supplied overrides (source_filters["pexels"])
    if extra_params:
        params.update(extra_params)

    def call() -> list[dict]:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params=params,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("photos", [])

    log.info("Pexels search images  q=%r per_page=%d extra=%s", q, per_page, extra_params)
    result = _with_backoff(call, backoff)
    _search_cache[key] = result
    return result


def _pexels_pick_image_url(photo: dict) -> Optional[str]:
    src = photo.get("src") or {}
    return src.get("large") or src.get("original") or src.get("medium")


# ---------------------------------------------------------------------------
# Pexels — videos
# ---------------------------------------------------------------------------

def _pexels_search_videos(
    api_key: str,
    query: str,
    per_page: int,
    backoff: list[int],
    extra_params: dict | None = None,
) -> list[dict]:
    key = ("pexels", "video", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache:
        return _search_cache[key]

    headers = {"Authorization": api_key}
    q = " ".join(query.splitlines())

    params: dict = {
        "query":    q,
        "per_page": per_page,
    }
    if extra_params:
        params.update(extra_params)

    def call() -> list[dict]:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params=params,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("videos", [])

    log.info("Pexels search videos  q=%r per_page=%d extra=%s", q, per_page, extra_params)
    result = _with_backoff(call, backoff)
    _search_cache[key] = result
    return result


def _pexels_pick_video_url(video: dict) -> Optional[str]:
    files = video.get("video_files") or []
    mp4s  = [
        f for f in files
        if (f.get("file_type") or "").lower() == "video/mp4" and f.get("link")
    ]
    if not mp4s:
        return None
    mp4s.sort(key=lambda f: -(f.get("width") or 0))
    for f in mp4s:
        if (f.get("width") or 0) >= 720:
            return f["link"]
    return mp4s[0]["link"]


# ---------------------------------------------------------------------------
# Pixabay — images
# ---------------------------------------------------------------------------

def _pixabay_search_images(
    api_key: str,
    query: str,
    per_page: int,
    backoff: list[int],
    extra_params: dict | None = None,
) -> list[dict]:
    key = ("pixabay", "image", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache:
        return _search_cache[key]

    q = " ".join(query.splitlines())[:100]

    # Always-on Pixabay image defaults
    params: dict = {
        "key":         api_key,
        "q":           q,
        "image_type":  "photo",
        "orientation": "horizontal",
        "safesearch":  "true",
        "order":       "popular",
        "per_page":    per_page,
    }
    # Merge caller-supplied overrides (source_filters["pixabay"])
    if extra_params:
        params.update(extra_params)

    def call() -> list[dict]:
        r = requests.get(
            "https://pixabay.com/api/",
            params=params,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("hits", [])

    log.info("Pixabay search images  q=%r per_page=%d extra=%s", q, per_page, extra_params)
    result = _with_backoff(call, backoff)
    _search_cache[key] = result
    return result


def _pixabay_pick_image_url(hit: dict) -> Optional[str]:
    return hit.get("largeImageURL") or hit.get("webformatURL")


# ---------------------------------------------------------------------------
# Pixabay — videos
# ---------------------------------------------------------------------------

def _pixabay_search_videos(
    api_key: str,
    query: str,
    per_page: int,
    backoff: list[int],
    extra_params: dict | None = None,
) -> list[dict]:
    key = ("pixabay", "video", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache:
        return _search_cache[key]

    q = " ".join(query.splitlines())[:100]

    # Always-on Pixabay video defaults
    params: dict = {
        "key":        api_key,
        "q":          q,
        "video_type": "film",
        "safesearch": "true",
        "order":      "popular",
        "per_page":   per_page,
    }
    # Merge caller-supplied overrides (source_filters["pixabay"])
    if extra_params:
        params.update(extra_params)

    def call() -> list[dict]:
        r = requests.get(
            "https://pixabay.com/api/videos/",
            params=params,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("hits", [])

    log.info("Pixabay search videos  q=%r per_page=%d extra=%s", q, per_page, extra_params)
    result = _with_backoff(call, backoff)
    _search_cache[key] = result
    return result


def _pixabay_pick_video_url(hit: dict) -> Optional[str]:
    vids = hit.get("videos") or {}
    for tier in ("large", "medium", "small", "tiny"):
        v = vids.get(tier) or {}
        if v.get("url"):
            return v["url"]
    return None


# ---------------------------------------------------------------------------
# Preview URL picker helpers (Phase 1 — search/scoring resolution)
# ---------------------------------------------------------------------------

def _pexels_pick_preview_image_url(photo: dict) -> str | None:
    """Pick a preview-resolution image URL (~1200px wide) from a Pexels photo object."""
    src = photo.get("src") or {}
    return src.get("medium") or src.get("small") or src.get("large")


def _pexels_pick_preview_video_url(video: dict) -> tuple[str | None, int, int]:
    """Pick a preview-resolution video URL (target ≤1280px wide) from a Pexels video object.

    Rule (B3): largest width <= 1280; if none exists, smallest width > 1280.
    Do NOT use "smallest >= 720" — that picks 1080p/4K when those are the only options.
    Returns (url, width, height).
    """
    files = [f for f in (video.get("video_files") or [])
             if (f.get("file_type") or "").lower() == "video/mp4" and f.get("link")]
    files.sort(key=lambda f: f.get("width") or 0)
    candidates = [f for f in files if (f.get("width") or 0) <= 1280]
    if candidates:
        f = candidates[-1]  # largest that still fits
        return f["link"], f.get("width", 0), f.get("height", 0)
    if files:
        f = files[0]  # smallest above 1280 as last resort
        return f["link"], f.get("width", 0), f.get("height", 0)
    return None, 0, 0


def _pixabay_pick_preview_image_url(hit: dict) -> str | None:
    """Pick the best downloadable image URL from a Pixabay hit object.

    Prefer largeImageURL (≤1280px, same CDN as webformatURL) so CLIP scores are
    comparable with Pexels medium images (~1280px).  webformatURL (~640px) makes
    Pixabay images score lower even when content is equally relevant.
    """
    return hit.get("largeImageURL") or hit.get("webformatURL") or hit.get("previewURL")


def _pixabay_pick_preview_video_url(hit: dict) -> str | None:
    """Pick a preview-resolution video URL (small tier) from a Pixabay hit object."""
    vids = hit.get("videos") or {}
    for tier in ("small", "medium", "large", "tiny"):
        v = vids.get(tier) or {}
        if v.get("url"):
            return v["url"]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_images(
    search_prompt: str,
    n:             int,
    output_dir:    Path,
    api_keys:      dict,
    cfg:           dict,
    item:          dict | None = None,
) -> tuple[list[tuple[Path, dict]], list[dict]]:
    """
    Search and download up to n images into output_dir/<source>/.
    Returns a tuple of (list of (Path, info_dict), list of pending_candidates).

    Already-downloaded files (by filename) are reused without re-downloading.

    Parameters
    ----------
    search_prompt : str
        Fallback query used when item has no search_queries.
    n : int
        Total desired image count across all queries (deduplicated).
    output_dir : Path
        Base directory; sub-directories per source are created automatically.
    api_keys : dict
        Must contain "pexels" and/or "pixabay" keys.
    cfg : dict
        Server config dict (sources, backoff_seconds, rate_limits, …).
    item : dict | None
        Optional manifest item dict.  Recognised fields:
          - search_queries (list[str]): rotate through multiple queries
          - source_filters (dict):      per-source param overrides
    """
    if n <= 0:
        return [], []

    backoff  = cfg.get("backoff_seconds", [5, 15, 45])
    sources  = cfg.get("sources", ["pexels", "pixabay"])
    queries  = _get_queries(search_prompt, item, n)

    # Initialise thumbnail map on item so scorer.py thumbnail pass-1.5 can use it.
    # Maps str(local_dest_path) → thumbnail_url (smaller API-provided preview URL).
    if item is not None:
        item.setdefault("_thumbnails", {})

    def _fetch_images_from_source(source: str) -> tuple[list, dict, list]:
        """
        Search + download images from one source independently.

        Two-phase approach:
          Phase 1 — Run all queries concurrently (up to max_concurrent workers).
                     Each worker calls the API then jitters immediately, so jitter
                     only gates between successive search calls — not the full
                     search+download cycle.
          Phase 2 — Deduplicate hits across queries, apply budget, single download pass.
        """
        n_src = cfg.get("source_limits", {}).get(source, {}).get("candidates_images")
        if n_src is None:
            raise ValueError(f"candidates_images not set for source '{source}' — caller must supply source_limits")
        if n_src == 0:
            log.info("fetch_images: skipping source %s (candidates_images=0)", source)
            return [], {}, []

        src_dir = output_dir / source
        src_dir.mkdir(parents=True, exist_ok=True)
        source_filters = _get_source_filters(item, source)
        rl = cfg.get("rate_limits", {}).get(source, {})
        # Keyword-indexed sources get a per-source query list optimised for
        # short keyword phrases.  Pexels/Pixabay keep the shared query list.
        source_queries = (
            _get_queries_for_source(search_prompt, item, n_src, source)
            if source in _KEYWORD_INDEXED_SOURCES
            else queries
        )
        n_q_workers = min(len(source_queries), max(1, rl.get("max_concurrent", 1)))
        pexels_key = api_keys.get("pexels", "")

        # ------------------------------------------------------------------
        # Phase 1 — search all queries concurrently.
        # Each worker returns pre-built task dicts with "_uid" for dedup.
        # ------------------------------------------------------------------
        def _search_one(q_per_q: tuple) -> list[dict]:
            query, per_q = q_per_q
            # Keyword-indexed sources (wikimedia, europeana, openverse) use
            # title/full-text indexing and have variable license-pass rates.
            # Over-fetch by 3× so that after license filtering we still have
            # enough candidates to fill the n_src budget (20 / 40 / 60).
            # Pexels/Pixabay semantic search is already precise — no over-fetch.
            if source in _KEYWORD_INDEXED_SOURCES:
                fetch_n = min(per_q * 3, 500)
            else:
                fetch_n = min(per_q, n_src)
            pre: list[dict] = []
            try:
                if source == "pexels":
                    hits = _pexels_search_images(
                        pexels_key, query, fetch_n, backoff,
                        extra_params=source_filters or None,
                    )
                    for i, ph in enumerate(hits):
                        pid = str(ph.get("id", i));  uid = f"pexels_img_{pid}"
                        url = _pexels_pick_image_url(ph)
                        if not url: continue
                        dest = src_dir / f"{uid}.jpg"
                        src_ph = ph.get("src") or {}
                        thumb  = src_ph.get("medium") or src_ph.get("small") or src_ph.get("tiny")
                        info = {
                            "source_site":     "pexels",
                            "asset_page_url":  ph.get("url", ""),
                            "file_url":        url,
                            "preview_url":     _pexels_pick_preview_image_url(ph),
                            "title":           ph.get("alt") or _slug_to_title(ph.get("url", "")),
                            "description":     ph.get("alt", ""),
                            "tags":            [],
                            "photographer":    ph.get("photographer", ""),
                            "license_summary": "Pexels License",
                            "license_url":     "https://www.pexels.com/license/",
                            "query_used":      query,
                            "width":           ph.get("width", 0),
                            "height":          ph.get("height", 0),
                        }
                        dl_url = info.get("preview_url") or url
                        pre.append({"_uid": uid, "dest": dest, "url": dl_url, "info": info,
                                    "thumb": thumb, "headers": {"Authorization": pexels_key}})

                elif source == "pixabay":
                    hits = _pixabay_search_images(
                        api_keys.get("pixabay", ""), query, fetch_n, backoff,
                        extra_params=source_filters or None,
                    )
                    for i, hit in enumerate(hits):
                        hid = str(hit.get("id", i));  uid = f"pixabay_img_{hid}"
                        url = _pixabay_pick_image_url(hit)
                        if not url: continue
                        ext  = Path(url.split("?")[0]).suffix or ".jpg"
                        dest = src_dir / f"{uid}{ext}"
                        thumb = hit.get("webformatURL") or hit.get("previewURL")
                        info = {
                            "source_site":     "pixabay",
                            "asset_page_url":  hit.get("pageURL", ""),
                            "file_url":        url,
                            "preview_url":     _pixabay_pick_preview_image_url(hit),
                            "title":           _slug_to_title(hit.get("pageURL", "")),
                            "description":     "",
                            "tags":            [t.strip() for t in hit.get("tags", "").split(",") if t.strip()],
                            "photographer":    hit.get("user", ""),
                            "license_summary": "Pixabay Content License",
                            "license_url":     "https://pixabay.com/service/license-summary/",
                            "query_used":      query,
                            "width":           hit.get("imageWidth", 0),
                            "height":          hit.get("imageHeight", 0),
                        }
                        dl_url = info.get("preview_url") or url
                        pre.append({"_uid": uid, "dest": dest, "url": dl_url, "info": info, "thumb": thumb})

                elif source == "openverse":
                    candidates = _source_search_openverse_images(
                        api_keys, query, fetch_n, backoff, extra_params=source_filters or None,
                    )
                    # Keyword-indexed: descriptive phrases often return 0.  Retry with
                    # progressively shorter queries until we get results.
                    if not candidates:
                        for fb_q in _keyword_fallbacks(query, item):
                            candidates = _source_search_openverse_images(
                                api_keys, fb_q, fetch_n, backoff, extra_params=source_filters or None,
                            )
                            if candidates:
                                log.debug("openverse fallback q=%r -> %d results", fb_q, len(candidates))
                                break
                    for c in candidates:
                        sid = c.get("source_id", "");  uid = f"openverse_img_{sid}"
                        url = c["file_url"]
                        if not url: continue
                        preview = c.get("preview_url", "")
                        # Smart dl_url: prefer file_url (full-res) when its CDN is already
                        # in DOWNLOAD_ALLOWLIST (e.g. live.staticflickr.com, upload.wikimedia.org).
                        # For providers on non-allowlisted CDNs (thingiverse, museum servers, etc.),
                        # fall back to preview_url — api.openverse.org is always allowlisted and
                        # serves a usable thumbnail for every image.  Without this fallback,
                        # non-Flickr Openverse results silently go to pending_host_review and
                        # never save, causing the delivered count to drop to ~3 per item.
                        file_host = urlparse(url).hostname or ""
                        file_allowed = _is_host_allowed(file_host) in ("static", "dynamic")
                        dl_url = url if file_allowed else (preview or url)
                        ext = Path(dl_url.split("?")[0]).suffix or ".jpg"
                        if ext.lower() not in {".jpg", ".jpeg", ".png", ".gif"}: ext = ".jpg"
                        dest = src_dir / f"{uid}{ext}"
                        info  = {**c, "file_url": url, "preview_url": preview}
                        pre.append({"_uid": uid, "dest": dest, "url": dl_url, "info": info, "thumb": preview})

                elif source == "wikimedia":
                    candidates = _source_search_wikimedia_images(
                        api_keys, query, fetch_n, backoff, extra_params=source_filters or None,
                    )
                    # Keyword-indexed: retry with shorter queries on zero results.
                    if not candidates:
                        for fb_q in _keyword_fallbacks(query, item):
                            candidates = _source_search_wikimedia_images(
                                api_keys, fb_q, fetch_n, backoff, extra_params=source_filters or None,
                            )
                            if candidates:
                                log.debug("wikimedia fallback q=%r -> %d results", fb_q, len(candidates))
                                break
                    for c in candidates:
                        sid = c.get("source_id", "")
                        uid = f"wikimedia_img_{re.sub(r'[^a-zA-Z0-9_-]', '_', sid)}"
                        url = c["file_url"]
                        if not url: continue
                        mime = c.get("mime", "image/jpeg")
                        ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif"}.get(mime, ".jpg")
                        dest = src_dir / f"{uid}{ext}"
                        thumb = c.get("preview_url", "")
                        info  = {**c}
                        dl_url = c.get("preview_url") or url
                        pre.append({"_uid": uid, "dest": dest, "url": dl_url, "info": info, "thumb": thumb})

                elif source == "europeana":
                    candidates = _source_search_europeana_images(
                        api_keys, query, fetch_n, backoff, extra_params=source_filters or None,
                    )
                    # Keyword-indexed: retry with shorter queries on zero results.
                    if not candidates:
                        for fb_q in _keyword_fallbacks(query, item):
                            candidates = _source_search_europeana_images(
                                api_keys, fb_q, fetch_n, backoff, extra_params=source_filters or None,
                            )
                            if candidates:
                                log.debug("europeana fallback q=%r -> %d results", fb_q, len(candidates))
                                break
                    for c in candidates:
                        sid = c.get("source_id", "")
                        uid = f"europeana_img_{re.sub(r'[^a-zA-Z0-9_-]', '_', sid)}"
                        url = c["file_url"]
                        if not url: continue
                        ext = Path(url.split("?")[0]).suffix.lower()
                        if ext not in {".jpg", ".jpeg", ".png", ".gif"}: ext = ".jpg"
                        dest = src_dir / f"{uid}{ext}"
                        thumb = c.get("preview_url", "")
                        info  = {**c}
                        dl_url = c.get("preview_url") or url
                        pre.append({"_uid": uid, "dest": dest, "url": dl_url, "info": info, "thumb": thumb})

                elif source == "archive":
                    candidates = _source_search_archive_images(
                        api_keys, query, fetch_n, backoff, extra_params=source_filters or None,
                    )
                    for c in candidates:
                        sid = c.get("source_id", "");  uid = f"archive_img_{sid}"
                        url = c["file_url"]
                        if not url: continue
                        ext = Path(url.split("?")[0]).suffix.lower()
                        if ext not in {".jpg", ".jpeg", ".png", ".gif"}: ext = ".jpg"
                        dest = src_dir / f"{uid}{ext}"
                        thumb = c.get("preview_url", "")
                        info  = {**c}
                        dl_url = c.get("preview_url") or url
                        pre.append({"_uid": uid, "dest": dest, "url": dl_url, "info": info, "thumb": thumb})

            except Exception as exc:
                log.warning("fetch_images %s q=%r failed: %s", source, query, exc)

            _jitter(cfg, source)  # gate between search API calls, not between search+download cycles
            return pre

        all_pre: list[dict] = []
        with ThreadPoolExecutor(max_workers=n_q_workers) as qex:
            for batch in qex.map(_search_one, source_queries):
                all_pre.extend(batch)

        # ------------------------------------------------------------------
        # Phase 2 — deduplicate across queries, apply budget, download once.
        # ------------------------------------------------------------------
        seen_ids_src: set[str] = set()
        tasks: list[dict] = []
        for pt in all_pre:
            if len(tasks) >= n_src:
                break
            uid = pt.pop("_uid")
            if uid in seen_ids_src:
                continue
            seen_ids_src.add(uid)
            tasks.append(pt)

        src_saved, src_thumbs, src_pending = _run_download_tasks(tasks, cfg, label=f"img/{source}")
        return src_saved, src_thumbs, src_pending

    # Run all sources in parallel — each source does its own API search + file downloads
    # concurrently with every other source.  Within each source, _run_download_tasks
    # further parallelises individual file downloads via its own ThreadPoolExecutor.
    all_saved:   list[tuple[Path, dict]] = []
    all_thumbs:  dict[str, str]          = {}
    all_pending: list[dict]              = []
    n_src_workers = min(len(sources), cfg.get("source_workers", len(sources)))
    log.info("fetch_images: running %d source(s) in parallel (workers=%d)", len(sources), n_src_workers)
    with ThreadPoolExecutor(max_workers=n_src_workers) as ex:
        for src_saved, src_thumbs, src_pending in ex.map(_fetch_images_from_source, sources):
            all_saved.extend(src_saved)
            all_thumbs.update(src_thumbs)
            all_pending.extend(src_pending)

    if item is not None:
        item["_thumbnails"].update(all_thumbs)

    # Return all candidates (up to max_candidates_per_item) so every source contributes.
    max_total = cfg.get("max_candidates_per_item", len(all_saved))
    return all_saved[:max_total], all_pending


def fetch_videos(
    search_prompt: str,
    n:             int,
    output_dir:    Path,
    api_keys:      dict,
    cfg:           dict,
    item:          dict | None = None,
) -> tuple[list[tuple[Path, dict]], list[dict]]:
    """
    Search and download up to n videos into output_dir/<source>/.
    Returns a tuple of (list of (Path, info_dict), list of pending_candidates).

    Already-downloaded files are reused without re-downloading.

    Parameters
    ----------
    search_prompt : str
        Fallback query used when item has no search_queries.
    n : int
        Total desired video count across all queries (deduplicated).
    output_dir : Path
        Base directory; sub-directories per source are created automatically.
    api_keys : dict
        Must contain "pexels" and/or "pixabay" keys.
    cfg : dict
        Server config dict (sources, backoff_seconds, rate_limits, …).
    item : dict | None
        Optional manifest item dict.  Recognised fields:
          - search_queries (list[str]): rotate through multiple queries
          - source_filters (dict):      per-source param overrides
    """
    if n <= 0:
        return [], []

    backoff  = cfg.get("backoff_seconds", [5, 15, 45])
    sources  = cfg.get("sources", ["pexels", "pixabay"])
    # B3 for videos: do NOT inject location into regular queries.
    # Pexels returns 5000+ loosely-matched results for any query — location prefix
    # adds no signal there. Pixabay uses strict tag matching — location-only query
    # is added per-source below (Pixabay only) to capture all location-tagged videos.
    queries  = _get_queries(search_prompt, item, n, inject_location=False)

    # Extract location term for Pixabay-only location query
    _loc_term = ""
    if item:
        kws = item.get("include_keywords") or []
        _loc_term = next((kw for kw in kws if kw and kw[0].isupper() and len(kw) > 3), "")

    def _fetch_videos_from_source(source: str) -> tuple[list[tuple[Path, dict]], list[dict]]:
        """
        Search + download videos from one source independently.

        Two-phase approach (same as fetch_images):
          Phase 1 — Run all queries concurrently (up to max_concurrent workers).
                     Jitter fires immediately after each API call, not after downloads.
          Phase 2 — Deduplicate, apply budget, single download pass.

        candidates_videos is the TOTAL file budget for this source across ALL queries
        combined (including the Pixabay location query).
        """
        n_src = cfg.get("source_limits", {}).get(source, {}).get("candidates_videos")
        if n_src is None:
            raise ValueError(f"candidates_videos not set for source '{source}' — caller must supply source_limits")
        if n_src == 0:
            log.info("fetch_videos: skipping source %s (candidates_videos=0)", source)
            return [], []

        src_dir = output_dir / source
        src_dir.mkdir(parents=True, exist_ok=True)
        source_filters = _get_source_filters(item, source)

        # Keyword-indexed sources get per-source optimised queries.
        # Pixabay keeps its existing location-first override.
        # Pexels uses the shared descriptive queries unchanged.
        if source in _KEYWORD_INDEXED_SOURCES:
            source_queries = _get_queries_for_source(search_prompt, item, n_src, source)
        elif source == "pixabay" and _loc_term:
            # Prepend location-only query so location-tagged videos are captured first.
            source_queries = [(_loc_term, n_src)] + list(queries)
        else:
            source_queries = queries

        rl = cfg.get("rate_limits", {}).get(source, {})
        n_q_workers = min(len(source_queries), max(1, rl.get("max_concurrent", 1)))

        # ------------------------------------------------------------------
        # Phase 1 — search all queries concurrently.
        # ------------------------------------------------------------------
        def _search_one(q_per_q: tuple) -> list[dict]:
            query, per_q = q_per_q
            if source in _KEYWORD_INDEXED_SOURCES:
                fetch_n = min(per_q * 3, 500)
            else:
                fetch_n = min(per_q, n_src)
            pre: list[dict] = []
            try:
                if source == "pexels":
                    hits = _pexels_search_videos(
                        api_keys.get("pexels", ""), query, fetch_n, backoff,
                        extra_params=source_filters or None,
                    )
                    for i, vd in enumerate(hits):
                        vid = str(vd.get("id", i));  uid = f"pexels_vid_{vid}"
                        url = _pexels_pick_video_url(vd)
                        if not url: continue
                        dest = src_dir / f"{uid}.mp4"
                        preview_url_v, pw, ph_v = _pexels_pick_preview_video_url(vd)
                        info = {
                            "source_site":     "pexels",
                            "asset_page_url":  vd.get("url", ""),
                            "file_url":        url,
                            "preview_url":     preview_url_v,
                            "preview_width":   pw,
                            "preview_height":  ph_v,
                            "video_files":     [{"width": f.get("width", 0), "height": f.get("height", 0), "url": f["link"]}
                                                for f in (vd.get("video_files") or []) if f.get("link")],
                            "title":           _slug_to_title(vd.get("url", "")),
                            "description":     "",
                            "tags":            [],
                            "photographer":    (vd.get("user") or {}).get("name", ""),
                            "license_summary": "Pexels License",
                            "license_url":     "https://www.pexels.com/license/",
                            "query_used":      query,
                            "width":           vd.get("width", 0),
                            "height":          vd.get("height", 0),
                        }
                        dl_url = info.get("preview_url") or url
                        pre.append({"_uid": uid, "dest": dest, "url": dl_url, "info": info})

                elif source == "pixabay":
                    hits = _pixabay_search_videos(
                        api_keys.get("pixabay", ""), query, fetch_n, backoff,
                        extra_params=source_filters or None,
                    )
                    for i, hit in enumerate(hits):
                        hid = str(hit.get("id", i));  uid = f"pixabay_vid_{hid}"
                        url = _pixabay_pick_video_url(hit)
                        if not url: continue
                        dest = src_dir / f"{uid}.mp4"
                        vids_data = hit.get("videos") or {}
                        info = {
                            "source_site":     "pixabay",
                            "asset_page_url":  hit.get("pageURL", ""),
                            "file_url":        url,
                            "preview_url":     _pixabay_pick_preview_video_url(hit),
                            "video_files":     [{"tier": t, "width": (vids_data.get(t) or {}).get("width", 0),
                                                  "url": (vids_data.get(t) or {}).get("url", "")}
                                                for t in ("large", "medium", "small", "tiny")
                                                if (vids_data.get(t) or {}).get("url")],
                            "title":           _slug_to_title(hit.get("pageURL", "")),
                            "description":     "",
                            "tags":            [t.strip() for t in hit.get("tags", "").split(",") if t.strip()],
                            "photographer":    hit.get("user", ""),
                            "license_summary": "Pixabay Content License",
                            "license_url":     "https://pixabay.com/service/license-summary/",
                            "query_used":      query,
                            "width":           (vids_data.get("large") or vids_data.get("medium") or {}).get("width", 0),
                            "height":          (vids_data.get("large") or vids_data.get("medium") or {}).get("height", 0),
                        }
                        dl_url = info.get("preview_url") or url
                        pre.append({"_uid": uid, "dest": dest, "url": dl_url, "info": info})

                elif source == "wikimedia":
                    candidates = _source_search_wikimedia_videos(
                        api_keys, query, fetch_n, backoff, extra_params=source_filters or None,
                    )
                    for c in candidates:
                        sid = c.get("source_id", "")
                        uid = f"wikimedia_vid_{re.sub(r'[^a-zA-Z0-9_-]', '_', sid)}"
                        url = c["file_url"]
                        if not url: continue
                        mime = c.get("mime", "video/webm")
                        ext  = {"video/mp4": ".mp4", "video/webm": ".webm", "video/ogg": ".ogv"}.get(mime, ".webm")
                        dest = src_dir / f"{uid}{ext}"
                        info = {**c, "video_files": [{"url": url, "width": c.get("width", 0), "height": c.get("height", 0)}]}
                        pre.append({"_uid": uid, "dest": dest, "url": url, "info": info})

                elif source == "archive":
                    candidates = _source_search_archive_videos(
                        api_keys, query, fetch_n, backoff, extra_params=source_filters or None,
                    )
                    for c in candidates:
                        sid = c.get("source_id", "");  uid = f"archive_vid_{sid}"
                        url = c["file_url"]
                        if not url: continue
                        dest = src_dir / f"{uid}.mp4"
                        info = {**c, "video_files": [{"url": url, "width": c.get("width", 0), "height": c.get("height", 0)}]}
                        dl_url = c.get("preview_url") or url
                        pre.append({"_uid": uid, "dest": dest, "url": dl_url, "info": info})

            except Exception as exc:
                log.warning("fetch_videos %s q=%r failed: %s", source, query, exc)

            _jitter(cfg, source)  # gate between search API calls, not between search+download cycles
            return pre

        all_pre: list[dict] = []
        with ThreadPoolExecutor(max_workers=n_q_workers) as qex:
            for batch in qex.map(_search_one, source_queries):
                all_pre.extend(batch)

        # ------------------------------------------------------------------
        # Phase 2 — deduplicate across queries, apply budget, download once.
        # ------------------------------------------------------------------
        seen_ids_src: set[str] = set()
        tasks: list[dict] = []
        for pt in all_pre:
            if len(tasks) >= n_src:
                break
            uid = pt.pop("_uid")
            if uid in seen_ids_src:
                continue
            seen_ids_src.add(uid)
            tasks.append(pt)

        src_saved, _, src_pending = _run_download_tasks(tasks, cfg, label=f"vid/{source}")
        return src_saved, src_pending

    # Run all sources in parallel
    all_saved:   list[tuple[Path, dict]] = []
    all_pending: list[dict]              = []
    n_src_workers = min(len(sources), cfg.get("source_workers", len(sources)))
    log.info("fetch_videos: running %d source(s) in parallel (workers=%d)", len(sources), n_src_workers)
    with ThreadPoolExecutor(max_workers=n_src_workers) as ex:
        for src_saved, src_pending in ex.map(_fetch_videos_from_source, sources):
            all_saved.extend(src_saved)
            all_pending.extend(src_pending)

    # Return all candidates (up to max_candidates_per_item) so every source contributes.
    max_total = cfg.get("max_candidates_per_item", len(all_saved))
    return all_saved[:max_total], all_pending
