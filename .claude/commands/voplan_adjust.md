# voplan_adjust

Compare VO timelines in a VOPlan.en.json against actual speech timestamps
detected by Whisper in a rendered mp4.  Prints a 3-column table:
  Line | VOPlan timeline | Whisper detected timeline

Usage: /voplan_adjust <mp4_path> <voplan_path>

---

## Step 1 — Parse arguments

Split $ARGUMENTS on whitespace into exactly two parts:
  - `MP4_PATH`    — full path to the .mp4 file
  - `VOPLAN_PATH` — full path to the VOPlan.en.json file

If either is missing or the files do not exist on disk, print:
  `✗ Usage: /voplan_adjust <mp4_path> <voplan_path>`
and stop.

---

## Step 2 — Run this Python script exactly as written

```python
import difflib
import json
import re
import sys
import warnings
import whisper

warnings.filterwarnings("ignore")

MP4_PATH    = "__MP4_PATH__"
VOPLAN_PATH = "__VOPLAN_PATH__"

# ── 1. Whisper transcription ────────────────────────────────────────────────
print("  Running Whisper…", flush=True)
model  = whisper.load_model("base")
result = model.transcribe(MP4_PATH, word_timestamps=True, verbose=False)

whisper_segs = [
    {
        "text":  seg["text"].strip(),
        "start": seg["start"],
        "end":   seg["end"],
    }
    for seg in result["segments"]
    if seg["text"].strip()
]

# ── 2. Load VOPlan ──────────────────────────────────────────────────────────
with open(VOPLAN_PATH, encoding="utf-8") as f:
    voplan = json.load(f)

vo_items = voplan.get("vo_items", [])

# ── 3. Normalise helper ─────────────────────────────────────────────────────
def norm(text):
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ── 4. Match each vo_item to the best Whisper segment ──────────────────────
#   Strategy: for each vo_item, find the whisper segment with the highest
#   SequenceMatcher ratio (normalised text).  A segment may match at most
#   one vo_item (greedy, in vo_item order).
used_segs = set()

def best_match(vo_text, segs, used):
    ref = norm(vo_text)
    best_idx, best_ratio = None, 0.0
    for i, seg in enumerate(segs):
        if i in used:
            continue
        ratio = difflib.SequenceMatcher(None, ref, norm(seg["text"])).ratio()
        if ratio > best_ratio:
            best_ratio, best_idx = ratio, i
    # Accept match only if similarity ≥ 0.4
    if best_ratio >= 0.4:
        used.add(best_idx)
        return best_idx, best_ratio
    return None, 0.0

matches = []   # list of (vo_item, whisper_seg_or_None, ratio)
for item in vo_items:
    idx, ratio = best_match(item.get("text", ""), whisper_segs, used_segs)
    wseg = whisper_segs[idx] if idx is not None else None
    matches.append((item, wseg, ratio))

# ── 5. Print table ──────────────────────────────────────────────────────────
W_LINE   = 52
W_VPLAN  = 22
W_WHIS   = 22

def hline(l, m, r, f="─"):
    return l + f*(W_LINE+2) + m + f*(W_VPLAN+2) + m + f*(W_WHIS+2) + r

def row(line, vplan, whis):
    return f"│ {line:<{W_LINE}} │ {vplan:^{W_VPLAN}} │ {whis:^{W_WHIS}} │"

print()
print(hline("┌","┬","┐"))
print(row("Line", "VOPlan timeline", "Whisper detected"))
print(hline("├","┼","┤"))

matched_count = 0
for item, wseg, ratio in matches:
    text      = item.get("text", "")
    v_start   = item.get("start_sec")
    v_end     = item.get("end_sec")
    v_str     = f"{float(v_start):.3f} → {float(v_end):.3f}" if v_start is not None else "N/A"
    w_str     = f"{wseg['start']:.3f} → {wseg['end']:.3f}"   if wseg else "—"

    # Truncate line text if too long
    display = text if len(text) <= W_LINE else text[:W_LINE-1] + "…"
    print(row(display, v_str, w_str))
    if wseg:
        matched_count += 1

print(hline("└","┴","┘"))
print(f"\n  {matched_count} of {len(vo_items)} lines matched to Whisper detections  "
      f"({len(whisper_segs)} segments found in mp4)\n")

# ── 6. Suggested adjustments ────────────────────────────────────────────────
#
# Timeline chain rules:
#   • First item in a scene: start_sec == scene_heads[scene_id]
#     → to shift it, change scene_heads[scene_id]
#   • Any other item[i]: start_sec == item[i-1].end_sec + item[i-1].pause_after_ms/1000
#     → to shift item[i], change item[i-1].pause_after_ms
#
# We only emit suggestions for matched items (those with a Whisper timestamp).

scene_heads = voplan.get("scene_heads", {})

# Build a set of item_ids that are the first item in their scene.
# item_id pattern: vo-{scene_id}-NNN  — first item has the lowest NNN per scene.
from collections import defaultdict
scene_first = {}   # scene_id → item_id of first item
scene_of    = {}   # item_id  → scene_id
for item in vo_items:
    iid = item["item_id"]            # e.g. "vo-sc01-001"
    parts = iid.split("-")           # ["vo","sc01","001"]
    if len(parts) >= 3:
        scene_id = parts[1]          # "sc01"
        seq      = int(parts[2])     # 1
        scene_of[iid] = scene_id
        if scene_id not in scene_first or seq < int(scene_first[scene_id].split("-")[2]):
            scene_first[scene_id] = iid

# Index vo_items by item_id for prev-item lookup
item_by_id  = {item["item_id"]: item for item in vo_items}
item_id_list = [item["item_id"] for item in vo_items]

print("── Suggested Adjustments " + "─" * 55)
print("   (for matched lines only — unmatched lines are omitted)")
print()

any_suggestion = False
for item, wseg, ratio in matches:
    if wseg is None:
        continue
    any_suggestion = True

    iid      = item["item_id"]
    text     = item.get("text", "")
    v_start  = float(item.get("start_sec", 0))
    w_start  = wseg["start"]
    delta_ms = round((w_start - v_start) * 1000)
    direction = f"+{delta_ms}ms" if delta_ms >= 0 else f"{delta_ms}ms"

    scene_id    = scene_of.get(iid, "")
    is_first    = (scene_first.get(scene_id) == iid)
    display     = f'"{text}"' if len(text) <= 48 else f'"{text[:47]}…"'

    print(f"  {display}")
    print(f"    VOPlan start: {v_start:.3f}s  →  Whisper start: {w_start:.3f}s  "
          f"(delta: {direction})")

    if is_first:
        cur_head = scene_heads.get(scene_id, v_start)
        new_head = round(cur_head + (w_start - v_start), 3)
        print(f"    → Change scene_heads[{scene_id}]: {cur_head:.3f} → {new_head:.3f}")
    else:
        # Find the previous item
        idx = item_id_list.index(iid)
        prev_item = item_by_id[item_id_list[idx - 1]]
        prev_end  = float(prev_item.get("end_sec", 0))
        cur_pause = int(prev_item.get("pause_after_ms", 0))
        new_pause = round((w_start - prev_end) * 1000)
        warn      = "  ⚠ clamped to 0 (VO clip too long to reach target)" if new_pause < 0 else ""
        new_pause_display = max(0, new_pause)
        print(f"    → Change pause_after_ms on {prev_item['item_id']}: "
              f"{cur_pause}ms → {new_pause_display}ms{warn}")

    print()

if not any_suggestion:
    print("  (no matched lines — nothing to suggest)\n")

# ── 7. Cross-clip pause context ──────────────────────────────────────────────
#
# For the first matched item: if it has a predecessor NOT in this clip,
# print the formula for the cross-clip pause before this clip.
# For the last matched item: if it has a successor NOT in this clip,
# print the formula for the cross-clip pause after this clip.
#
# Formula (cross-clip pause):
#   pause_after_ms on PREV_ITEM
#     = (prev_clip_cut_end  − prev_speech_end)      ← tail of previous clip
#     + (this_speech_start  − this_clip_cut_start)  ← head of this clip
#
# Formula (scene_head adjustment, when this clip contains the first item):
#   new scene_heads[scene_id]
#     = current_scene_head + (this_speech_start − this_clip_cut_start)

matched_items = [(item, wseg) for item, wseg, ratio in matches if wseg is not None]

if matched_items:
    first_item, first_wseg = matched_items[0]
    last_item,  last_wseg  = matched_items[-1]

    first_iid  = first_item["item_id"]
    last_iid   = last_item["item_id"]
    first_idx  = item_id_list.index(first_iid)
    last_idx   = item_id_list.index(last_iid)

    # Determine which adjacent items were NOT matched in this clip
    prev_iid = item_id_list[first_idx - 1] if first_idx > 0 else None
    next_iid = item_id_list[last_idx + 1]  if last_idx < len(item_id_list) - 1 else None

    matched_iids = {item["item_id"] for item, wseg, _ in matches if wseg is not None}

    has_cross_before = prev_iid and prev_iid not in matched_iids
    has_cross_after  = next_iid and next_iid not in matched_iids

    if has_cross_before or has_cross_after:
        print()
        print("── Cross-clip pause context " + "─" * 52)
        print()

    if has_cross_before:
        sp_start = first_wseg["start"]
        scene_id_first = scene_of.get(first_iid, "")
        is_scene_first = (scene_first.get(scene_id_first) == first_iid)

        if is_scene_first:
            cur_head = scene_heads.get(scene_id_first, float(first_item.get("start_sec", 0)))
            print(f"  SCENE HEAD ({first_iid} is first in scene):")
            print(f"    Speech starts at {sp_start:.3f}s in this clip")
            print(f"    new scene_heads[{scene_id_first}]")
            print(f"      = {cur_head:.3f} + ({sp_start:.3f} − this_clip_cut_start)")
            print(f"      = {cur_head:.3f} + ?  ← provide: this_clip_cut_start")
        else:
            print(f"  GAP BEFORE this clip  ({prev_iid} → {first_iid}):")
            print(f"    {first_iid} speech starts at {sp_start:.3f}s in this clip")
            print(f"    pause_after_ms on {prev_iid}")
            print(f"      = (prev_clip_cut_end − <{prev_iid} speech end from prev run>)")
            print(f"      + ({sp_start:.3f} − this_clip_cut_start)")
            print(f"    Provide: prev_clip_cut_end, this_clip_cut_start")
        print()

    if has_cross_after:
        sp_end = last_wseg["end"]
        print(f"  GAP AFTER this clip  ({last_iid} → {next_iid}):")
        print(f"    {last_iid} speech ends at {sp_end:.3f}s in this clip")
        print(f"    pause_after_ms on {last_iid}")
        print(f"      = (this_clip_cut_end − {sp_end:.3f})")
        print(f"      + (<{next_iid} speech start from next run> − next_clip_cut_start)")
        print(f"    Provide: this_clip_cut_end, next_clip_cut_start")
        print()
```

Before running, replace the two placeholder strings in the script:
  - `__MP4_PATH__`    → the actual value of MP4_PATH from Step 1
  - `__VOPLAN_PATH__` → the actual value of VOPLAN_PATH from Step 1

Then run the script using the project Python interpreter:
  `/home/tnnd/.virtualenvs/pipe/bin/python3`

Print the output exactly as produced — no reformatting.

---

## Step 3 — Fill in known values from conversation context

After printing the script output, check if any cross-clip pause formulas were printed.

For each formula that references `<{item_id} speech end/start from prev/next run>`:
- Search this conversation for a previous `/voplan_adjust` run that detected that item
- If found, substitute the actual Whisper speech_end or speech_start value into the formula
- Print the updated formula with the known value filled in

For each formula that still has unknowns (`this_clip_cut_start`, `this_clip_cut_end`,
`prev_clip_cut_end`, `next_clip_cut_start`):
- Ask the user to provide those cut points
- Once provided, compute the final pause_after_ms value and print:

  ```
  pause_after_ms on {item_id}: {current_ms}ms → {new_ms}ms  ◀
  ```
