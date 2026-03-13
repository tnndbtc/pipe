#!/usr/bin/env python3
# =============================================================================
# audio_pipeline_utils.py — Shared audio constants and helpers
# =============================================================================
#
# FIX 1  : Single authoritative source for duck-level constants.
# FIX 2A : resolve_duck_db() — consistent duck_db resolution used by both
#           the preview pipeline (music_review_pack.py) and the render
#           pipeline (gen_render_plan.py / render_video.py).
# FIX 8  : compute_vo_intervals() + apply_duck_envelope_numpy() so that
#           preview and render share identical ducking logic.
#
# All functions are pure (no file I/O, no subprocess) so they can be
# imported freely without side-effects.
# =============================================================================

import numpy as np

# ── FIX 1: Authoritative duck-level constants ──────────────────────────────
#
# These values mirror what render_video.py and apply_music_plan.py define
# locally.  Any module that previously had its own copy should import from
# here instead.
#
# render_video.py  line 77 : BASE_MUSIC_DB = -6.0
# render_video.py  line 656: duck_db default -12.0
# apply_music_plan.py line 36 : DEFAULT_DUCK_DB = -6.0   (legacy; kept in sync)
# music_review_pack.py line 37: DEFAULT_DUCK_DB = -6.0 / BASE_MUSIC_DB = -6.0

DEFAULT_DUCK_DB = -12.0   # matches render_video.py line 656 — reference value
BASE_MUSIC_DB   = -6.0    # un-ducked music level (mirrors render_video.py + apply_music_plan.py)

# Per-genre duck depth (copied verbatim from apply_music_plan.py DUCK_DB_BY_TYPE).
# Negative = quieter relative to BASE_MUSIC_DB.
DUCK_DB_BY_TYPE: dict[str, float] = {
    "ambient":    -5.0,
    "drone":      -5.0,
    "piano":      -7.0,
    "soft":       -7.0,
    "orchestral": -9.0,
    "epic":       -9.0,
}


# ── FIX 2A: Consistent duck_db resolution ─────────────────────────────────

def resolve_duck_db(music_item: dict, track_type: str | None = None) -> float:
    """
    Consistent duck_db resolution used by preview AND render.

    Resolution priority:
      1. Explicit ``duck_db`` key on ``music_item`` → use it (shot override wins).
      2. ``track_type`` lookup in DUCK_DB_BY_TYPE → use genre-specific depth.
         ``track_type`` is taken from the argument first, then from
         ``music_item["track_type"]`` if the argument is None.
      3. DEFAULT_DUCK_DB fallback (-12.0).

    Parameters
    ----------
    music_item : dict
        A music manifest item (may contain ``duck_db`` and/or ``track_type``).
    track_type : str | None
        Optional caller-supplied track type that overrides the item's own
        ``track_type`` field.  Pass None to use the item's field.

    Returns
    -------
    float
        The resolved duck depth in dB (always ≤ 0.0).

    Examples
    --------
    >>> resolve_duck_db({"duck_db": -8.0})
    -8.0
    >>> resolve_duck_db({"track_type": "piano"})
    -7.0
    >>> resolve_duck_db({})
    -12.0
    """
    # 1. Explicit override on the item
    if "duck_db" in music_item:
        return float(music_item["duck_db"])

    # 2. Genre lookup — caller-supplied track_type takes precedence over item field
    resolved_type = track_type if track_type is not None else music_item.get("track_type", "")
    if resolved_type and resolved_type in DUCK_DB_BY_TYPE:
        return DUCK_DB_BY_TYPE[resolved_type]

    # 3. Global default
    return DEFAULT_DUCK_DB


# ── FIX 8A: VO interval computation ───────────────────────────────────────

def compute_vo_intervals(
    shot: dict,
    vo_items: list[dict],
    pause_after_ms: float = 300.0,
    scene_tails: dict | None = None,
) -> list[tuple[float, float]]:
    """
    Compute shot-relative (t0, t1) second intervals where VO is active.

    This is a pure reimplementation of gen_render_plan.compute_duck_intervals_from_vo()
    combined with the vo_lines cursor logic in gen_render_plan.build_shot().
    It must produce identical duck interval endpoints to the render pipeline.

    How it works
    ------------
    1. Walk ``vo_items`` in order (same as build_shot does with vo_item_ids).
    2. Assign each item a shot-relative start/end time using a cursor:
           timeline_in_ms  = cursor_ms
           timeline_out_ms = cursor_ms + wav_dur_ms
           cursor_ms       += wav_dur_ms + item_pause_ms
       where ``wav_dur_ms = (end_sec - start_sec) * 1000`` from the item and
       ``item_pause_ms = item.get("pause_after_ms", pause_after_ms)``.
    3. For each placed line, produce a duck interval:
           t0 = max(0, timeline_in_ms  - fade_ms) / 1000
           t1 =       (timeline_out_ms + fade_ms) / 1000
       using ``fade_ms = round(fade_sec * 1000)`` extracted from the shot's
       music entry (caller should pass it via ``shot["music_fade_sec"]`` or
       this function uses 150 ms as the same default as gen_render_plan).
    4. Merge overlapping intervals (same algorithm as gen_render_plan).

    Parameters
    ----------
    shot : dict
        A ShotList shot dict.  May contain ``audio_intent.music_item_id`` and
        ``music_fade_sec`` (seconds).  Used only to extract fade_ms.
    vo_items : list[dict]
        Ordered VO manifest items for this shot.  Each must carry
        ``start_sec`` and ``end_sec`` (WAV timing from post_tts_analysis).
        Items that are placeholders (``is_placeholder=True``) or lack
        ``start_sec``/``end_sec`` are skipped, mirroring build_shot().
    pause_after_ms : float
        Default inter-line pause in milliseconds (300 ms = INTER_LINE_PAUSE_MS
        in gen_render_plan.py).  Overridden per-item by ``item["pause_after_ms"]``.
    scene_tails : dict | None
        Unused by this function; accepted for API symmetry with build_shot().

    Returns
    -------
    list[tuple[float, float]]
        Sorted, non-overlapping list of (t0, t1) in shot-relative seconds.
        Empty list when there are no placed VO items.

    Notes
    -----
    * Deduplication of identical (speaker, text) pairs is NOT performed here
      because that requires the full shot context.  Callers that need it should
      deduplicate ``vo_items`` before calling this function.
    * The fade_ms used for interval padding comes from ``shot.get("music_fade_sec", 0.15)``.
    """
    # Resolve fade_ms: use the shot's music_fade_sec if available, else 150 ms
    # (gen_render_plan default: fade_sec = music_item.get("fade_sec", 0.15))
    fade_sec_default = 0.15
    fade_sec = float(shot.get("music_fade_sec", fade_sec_default))
    fade_ms  = round(fade_sec * 1000)

    cursor_ms = 0
    raw: list[tuple[float, float]] = []

    for vo_item in vo_items:
        # Skip placeholders — mirrors: if not media_item or media_item.get("is_placeholder", True)
        if vo_item.get("is_placeholder", False):
            continue

        start_sec = vo_item.get("start_sec")
        end_sec   = vo_item.get("end_sec")
        if start_sec is None or end_sec is None:
            continue

        # WAV duration for this clip (same as build_shot: wav_dur_ms)
        wav_dur_ms = round((end_sec - start_sec) * 1000)
        if wav_dur_ms <= 0:
            continue

        # Shot-relative placement using cursor (identical to build_shot)
        timeline_in_ms  = cursor_ms
        timeline_out_ms = cursor_ms + wav_dur_ms
        item_pause_ms   = float(vo_item.get("pause_after_ms", pause_after_ms))
        cursor_ms       = timeline_out_ms + item_pause_ms

        # Duck interval for this line (same formula as compute_duck_intervals_from_vo)
        t0 = max(0.0, (timeline_in_ms  - fade_ms) / 1000.0)
        t1 =          (timeline_out_ms + fade_ms) / 1000.0
        raw.append((t0, t1))

    if not raw:
        return []

    # Merge overlapping intervals — same algorithm as gen_render_plan
    sorted_ivs = sorted(raw, key=lambda x: x[0])
    merged: list[list[float]] = [list(sorted_ivs[0])]
    for t0, t1 in sorted_ivs[1:]:
        if t0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])

    return [(round(a, 3), round(b, 3)) for a, b in merged]


# ── FIX 8B: Numpy duck envelope ────────────────────────────────────────────

def apply_duck_envelope_numpy(
    audio: "np.ndarray",
    sr: int,
    duck_intervals: list[tuple[float, float]],
    base_db: float,
    duck_db: float,
    fade_sec: float = 0.05,
) -> "np.ndarray":
    """
    Apply a per-VO-interval ducking amplitude envelope to a float32 numpy buffer.

    This mirrors build_duck_expr() from render_video.py in numpy space so that
    the music preview (music_review_pack.py) sounds identical to the rendered
    output (render_video.py).

    Envelope shape (mirrors build_duck_expr exactly)
    -------------------------------------------------
    The interval [t0, t1] already contains the fade padding (as written by
    gen_render_plan.compute_duck_intervals_from_vo).  Within the interval:

      t0 … t0+fade_sec  : linear ramp  base_amp → duck_amp  (fade-in / duck down)
      t0+fade_sec … t1-fade_sec : hold at duck_amp
      t1-fade_sec … t1  : linear ramp  duck_amp → base_amp  (fade-out / duck up)

    Outside all intervals: hold at base_amp.

    Amplitude definitions (same as build_duck_expr)
    ------------------------------------------------
      base_amp = 10 ^ (base_db / 20)
      duck_amp = base_amp × 10 ^ (duck_db / 20)

    ``duck_db`` is RELATIVE attenuation (negative = quieter).  For example:
      base_db=-6, duck_db=-12  →  duck_amp = 10^(-6/20) × 10^(-12/20)
                                           = 10^(-18/20)  ≈ 0.126  (-18 dB total)

    Parameters
    ----------
    audio : np.ndarray, shape (N,) or (N, C), dtype float32 or float64
        Input audio buffer.  Mono 1-D or multi-channel 2-D.  Modified in-place
        if the array is writable; otherwise a copy is returned.
    sr : int
        Sample rate in Hz (must match the time coordinates in duck_intervals).
    duck_intervals : list[tuple[float, float]]
        Sorted, non-overlapping (t0, t1) pairs in seconds — typically the
        output of compute_vo_intervals().  May be empty (returns audio unchanged).
    base_db : float
        Un-ducked music level in dB (typically BASE_MUSIC_DB = -6.0).
    duck_db : float
        Ducking depth relative to base_db in dB (negative, e.g. -12.0).
    fade_sec : float
        Duration of each linear ramp in seconds (default 0.05 s = 50 ms).
        Clamped per-interval so it never exceeds half the interval length.

    Returns
    -------
    np.ndarray
        Audio with the ducking envelope applied.  Same shape and dtype as input.

    Notes
    -----
    * The envelope is built at sample resolution; no FFmpeg filter-graph is
      involved.
    * The function works on float32 and float64 arrays.  Integer arrays will
      raise a TypeError — convert before calling.
    * If ``duck_intervals`` is empty, the audio is still scaled by ``base_amp``
      so the output level matches what build_duck_expr() would produce for a
      track with no ducking.
    """
    n_samples = audio.shape[0]
    base_amp  = 10.0 ** (base_db  / 20.0)
    duck_amp  = base_amp * (10.0 ** (duck_db / 20.0))

    # Start with the un-ducked level for every sample
    envelope = np.full(n_samples, base_amp, dtype=np.float64)

    for t0, t1 in duck_intervals:
        # Convert times to sample indices, clamp to buffer bounds
        s0 = max(0, int(t0 * sr))
        s1 = min(n_samples, int(t1 * sr))
        if s0 >= s1:
            continue

        interval_len = s1 - s0
        # Clamp fade length so it never exceeds half the interval (mirrors build_duck_expr)
        fade_samples = min(int(fade_sec * sr), interval_len // 2)

        if fade_samples > 0:
            # Ramp down: base_amp → duck_amp  over [s0, s0+fade_samples)
            envelope[s0 : s0 + fade_samples] = np.linspace(
                base_amp, duck_amp, fade_samples, dtype=np.float64
            )

        # Hold: duck_amp over [s0+fade_samples, s1-fade_samples)
        hold_start = s0 + fade_samples
        hold_end   = s1 - fade_samples
        if hold_end > hold_start:
            envelope[hold_start:hold_end] = duck_amp

        if fade_samples > 0:
            # Ramp up: duck_amp → base_amp  over [s1-fade_samples, s1)
            envelope[s1 - fade_samples : s1] = np.linspace(
                duck_amp, base_amp, fade_samples, dtype=np.float64
            )

    # Apply envelope: broadcast over channels if audio is 2-D (N, C)
    audio = audio * (
        envelope[:, np.newaxis] if audio.ndim == 2 else envelope
    ).astype(audio.dtype)

    return audio
