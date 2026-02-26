#!/usr/bin/env python3
# =============================================================================
# tag_music.py — Auto-tag music files with CLAP-derived mood descriptors
# =============================================================================
#
# Reads MP3/WAV/FLAC files, runs CLAP zero-shot classification against a
# curated mood vocabulary, and writes the top tags to the file's TXXX:MOOD
# ID3 field (or MOOD Vorbis comment for FLAC/OGG).
#
# Run ONCE per file.  gen_music_clip.py then reads these tags to match each
# music_item's music_mood to the best source track.
#
# Usage:
#   python tag_music.py                                   # scan current directory
#   python tag_music.py track.mp3                         # single file
#   python tag_music.py a.mp3 b.wav c.flac                # multiple files
#   python tag_music.py --dir projects/resources/music/   # scan specific directory
#   python tag_music.py --dry-run                         # show tags, don't write
#   python tag_music.py --top 12                          # keep 12 tags (default: 10)
#   python tag_music.py --force                           # re-tag already-tagged files
#
# Requirements (CPU-only — no GPU needed):
#   pip install laion-clap mutagen librosa soundfile numpy scipy torchvision
#
# CLAP checkpoint (~400 MB) is downloaded automatically to
#   ~/.cache/laion_clap/  on first run.
# =============================================================================

import argparse
import gc
import os
import sys
from contextlib import contextmanager
from pathlib import Path

# Silence huggingface_hub "unauthenticated requests" notice — no token needed.
# Must be set before laion_clap is imported.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import numpy as np

SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

CKPT_URL      = ("https://huggingface.co/lukewys/laion_clap/resolve/main/"
                 "music_audioset_epoch_15_esc_90.14.pt")
CKPT_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

CLAP_SR      = 48_000
CHUNK_SEC    = 10          # CLAP maximum receptive field
TEXT_BATCH   = 16          # text embeddings per forward pass (limits RAM)

# ── Mood vocabulary ───────────────────────────────────────────────────────────
#
# Tuned for narrative / cinematic music — mirrors the language that Stage 4
# uses in ShotList.audio_intent.music_mood.  Add or remove terms freely.
#
MOOD_VOCAB = [
    # Emotional character
    "dark",          "ominous",        "tense",          "dread",
    "suspenseful",   "unsettling",     "foreboding",     "eerie",
    "haunting",      "mysterious",     "supernatural",   "otherworldly",
    "ancient",       "ceremonial",     "ritual",         "reverent",
    "solemn",        "epic",           "triumphant",     "heroic",
    "majestic",      "powerful",       "hopeful",        "uplifting",
    "peaceful",      "serene",         "melancholic",    "sad",
    "sorrowful",     "wistful",        "aggressive",     "fierce",
    "urgent",        "military",
    # Energy / texture
    "low energy",    "high energy",    "building",       "crescendo",
    "fading",        "pulse",          "drone",          "minimal",
    "sparse",        "dense",          "rhythmic",       "driving",
    "steady",        "near silence",   "silence",        "static",
    "atmospheric",   "ambient",        "vast",           "hollow",
    # Instrumentation
    "drums",         "low drums",      "deep drums",     "percussion",
    "strings",       "cello",          "violin",         "viola",
    "brass",         "horn",           "trumpet",
    "piano",         "keys",
    "choir",         "choral",         "chanting",       "vocals",
    "orchestral",    "full orchestra",
    "bass",          "sub-bass",       "low frequency",
    # Tempo
    "slow",          "moderate",       "fast",
]


# ── Silence suppressor ────────────────────────────────────────────────────────

@contextmanager
def suppress_output():
    """
    Suppress stdout and stderr at the OS file-descriptor level.
    Required because tqdm and C extensions bypass Python's sys.stderr
    and write directly to fd 1 / fd 2.

    Flushes Python's internal buffers both on entry AND in the finally
    block (while fds still point to /dev/null) so block-buffered content
    never leaks to the restored real fds.
    """
    with open(os.devnull, "w") as devnull:
        devnull_fd  = devnull.fileno()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            yield
        finally:
            # Flush buffered content to /dev/null BEFORE restoring real fds,
            # otherwise block-buffered prints inside the context leak out.
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_clap(ckpt_dir: Path):
    """Load CLAP model, auto-downloading checkpoint if needed."""
    import laion_clap

    ckpt_path = ckpt_dir / CKPT_FILENAME
    if not ckpt_path.exists():
        print(f"[CLAP] Checkpoint not found — downloading to {ckpt_path} …")
        import urllib.request
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CKPT_URL, ckpt_path)
        print("[CLAP] Download complete.")

    print(f"[CLAP] Loading model …", end=" ", flush=True)
    # suppress per-weight "Loaded" spam, tqdm progress, and torch UserWarnings
    with suppress_output():
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        model.load_ckpt(str(ckpt_path))
    model.eval()
    print("ready.\n")
    return model


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio_48k(path: Path) -> np.ndarray:
    """Load any audio file as mono float32 at 48 kHz."""
    import librosa
    audio, _ = librosa.load(str(path), sr=CLAP_SR, mono=True)
    return audio.astype(np.float32)


def embed_file(model, audio: np.ndarray) -> np.ndarray:
    """
    Return a single (512,) embedding representing the whole file.
    Processes ONE 10-second chunk at a time to stay within RAM budget,
    then returns the mean across all chunks.
    """
    import torch

    chunk_frames = CHUNK_SEC * CLAP_SR
    chunk_embs   = []

    for start in range(0, len(audio), chunk_frames):
        chunk = audio[start: start + chunk_frames]
        if len(chunk) < chunk_frames:
            chunk = np.pad(chunk, (0, chunk_frames - len(chunk)))

        with torch.no_grad():
            emb = model.get_audio_embedding_from_data([chunk], use_tensor=False)

        chunk_embs.append(emb[0].copy())   # copy out before tensor is freed

    return np.mean(chunk_embs, axis=0)     # (512,)


# ── Tag computation ───────────────────────────────────────────────────────────

def compute_tags(model, audio: np.ndarray, top_n: int) -> list[tuple[str, float]]:
    """
    Score all vocab terms against the audio; return top_n (tag, score) pairs.
    Text vocab is embedded in small batches (TEXT_BATCH) to limit peak RAM.
    """
    import torch

    audio_emb = embed_file(model, audio)   # (512,)

    # Embed vocab in batches to avoid materialising all 80 text embeddings at once
    all_text_embs = []
    for i in range(0, len(MOOD_VOCAB), TEXT_BATCH):
        batch = MOOD_VOCAB[i: i + TEXT_BATCH]
        with torch.no_grad():
            embs = model.get_text_embedding(batch, use_tensor=False)
        all_text_embs.append(embs)

    text_embs = np.vstack(all_text_embs)   # (V, 512)
    scores    = text_embs @ audio_emb      # (V,)
    ranked    = sorted(zip(MOOD_VOCAB, scores.tolist()), key=lambda x: -x[1])
    return ranked[:top_n]


# ── Tag I/O ───────────────────────────────────────────────────────────────────

def read_mood_tag(path: Path) -> str | None:
    """Return existing TXXX:MOOD value, or None if absent."""
    try:
        suf = path.suffix.lower()
        if suf == ".mp3":
            from mutagen.id3 import ID3, ID3NoHeaderError
            try:
                txxx = ID3(str(path)).get("TXXX:MOOD")
                return str(txxx.text[0]) if txxx else None
            except ID3NoHeaderError:
                return None
        elif suf == ".flac":
            from mutagen.flac import FLAC
            vals = FLAC(str(path)).get("MOOD")
            return vals[0] if vals else None
        elif suf == ".ogg":
            from mutagen.oggvorbis import OggVorbis
            vals = OggVorbis(str(path)).get("MOOD")
            return vals[0] if vals else None
    except Exception:
        pass
    return None


def write_mood_tag(path: Path, tags: list[tuple[str, float]], dry_run: bool) -> str:
    """
    Write comma-separated top tags to the file's MOOD metadata field.
    Returns the tag string (for the summary line).
    """
    tag_str = ", ".join(t for t, _ in tags)

    if dry_run:
        return tag_str

    suf = path.suffix.lower()
    try:
        if suf == ".mp3":
            from mutagen.id3 import ID3, TXXX, ID3NoHeaderError
            try:
                id3 = ID3(str(path))
            except ID3NoHeaderError:
                id3 = ID3()
            id3.delall("TXXX:MOOD")
            id3.add(TXXX(encoding=3, desc="MOOD", text=tag_str))
            id3.save(str(path))
        elif suf == ".flac":
            from mutagen.flac import FLAC
            f = FLAC(str(path))
            f["MOOD"] = tag_str
            f.save()
        elif suf == ".ogg":
            from mutagen.oggvorbis import OggVorbis
            f = OggVorbis(str(path))
            f["MOOD"] = tag_str
            f.save()
        else:
            print(f"  [WARN] Tag writing not supported for {suf} — "
                  f"tags not saved.")
    except Exception as exc:
        print(f"  [ERROR] Could not write tag: {exc}", file=sys.stderr)

    return tag_str


# ── File collection ───────────────────────────────────────────────────────────

def collect_files(args) -> list[Path]:
    """Return list of audio files from positional args, --dir, or cwd scan."""
    if args.files:
        paths   = [Path(f) for f in args.files]
        missing = [p for p in paths if not p.exists()]
        if missing:
            for p in missing:
                print(f"[ERROR] File not found: {p}", file=sys.stderr)
            sys.exit(1)
        return paths

    scan_dir = Path(args.dir).resolve() if args.dir else Path.cwd()
    found    = sorted(
        p for p in scan_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not found:
        print(f"[WARN] No audio files found in {scan_dir}")
    return found


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-tag music files with CLAP-derived mood descriptors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python tag_music.py                               # scan cwd\n"
            "  python tag_music.py track.mp3                     # single file\n"
            "  python tag_music.py --dir projects/resources/music # specific dir\n"
            "  python tag_music.py --dry-run                     # preview only\n"
        ),
    )
    p.add_argument("files", nargs="*", metavar="FILE",
                   help="Audio file(s) to tag. Omit to scan current directory.")
    p.add_argument("--dir", metavar="DIR",
                   help="Directory to scan (default: current working directory).")
    p.add_argument("--top", type=int, default=10, metavar="N",
                   help="Number of top tags to keep per file (default: 10).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print tags without writing to files.")
    p.add_argument("--force", action="store_true",
                   help="Re-tag files that already have a MOOD tag.")
    p.add_argument("--ckpt-dir", default=None, metavar="DIR",
                   help="Directory for CLAP checkpoint "
                        "(default: ~/.cache/laion_clap/).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    files    = collect_files(args)
    if not files:
        sys.exit(0)

    ckpt_dir = (Path(args.ckpt_dir) if args.ckpt_dir
                else Path.home() / ".cache" / "laion_clap")
    model    = load_clap(ckpt_dir)

    total        = len(files)
    skipped      = 0
    tagged_count = 0
    error_count  = 0

    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{total}] {path.name}")

        if not args.force:
            existing = read_mood_tag(path)
            if existing:
                print(f"  [SKIP] Already tagged: {existing}\n")
                skipped += 1
                continue

        try:
            audio    = load_audio_48k(path)
            duration = len(audio) / CLAP_SR
            print(f"  Duration : {duration:.1f}s  "
                  f"({len(audio) // (CHUNK_SEC * CLAP_SR) + 1} chunks)")
            print(f"  Scoring  : {len(MOOD_VOCAB)} vocabulary terms …", flush=True)

            top_tags = compute_tags(model, audio, args.top)
            tag_str  = write_mood_tag(path, top_tags, args.dry_run)

            # ── Clear per-file summary line ───────────────────────────────────
            action = "[DRY-RUN]" if args.dry_run else "✓ tagged"
            print(f"  {action}  {path.name}  →  {tag_str}\n")

            tagged_count += 1

        except Exception as exc:
            print(f"  [ERROR] {exc}\n", file=sys.stderr)
            import traceback
            traceback.print_exc()
            error_count += 1

        finally:
            # Release audio array and torch caches between files
            try:
                del audio
            except NameError:
                pass
            gc.collect()

    print("─" * 60)
    print(f"  Tagged  : {tagged_count}")
    print(f"  Skipped : {skipped}  (use --force to re-tag)")
    print(f"  Errors  : {error_count}")
    print("✓ Done.")


if __name__ == "__main__":
    main()
