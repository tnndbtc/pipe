# list-top-voices

List top-5 Azure TTS voices by style count for four categories:
en-US Female, en-US Male, zh Female, zh Male.

---

## Step 1 — Run this Python script

```python
import json, textwrap

with open("projects/resources/azure_tts/index.json") as f:
    data = json.load(f)

voices = data["voices"]

results = []
for voice_name, v in voices.items():
    clips = v.get("clips", {})
    styles = [s for s in clips.keys() if s != ""]
    results.append({
        "name": voice_name,
        "locale": v.get("locale", ""),
        "locale_group": v.get("locale_group", ""),
        "gender": v.get("gender", ""),
        "style_count": len(styles),
        "styles": styles,
    })

results.sort(key=lambda x: -x["style_count"])

# Column widths (inner, excluding padding)
W_NUM   = 3
W_VOICE = 19
W_STYLE = 44

def pad(s, w): return s[:w].ljust(w)
def hline(l, m, r, f="─"):
    return l + f*(W_NUM+2) + m + f*(W_VOICE+2) + m + f*(W_STYLE+2) + r

def render_table(label, rows):
    print(f"  === {label} ===\n")
    print("  " + hline("┌","┬","┐"))
    print(f"  │ {pad('#',W_NUM)} │ {pad('Voice',W_VOICE)} │ {pad('Styles',W_STYLE)} │")
    print("  " + hline("├","┼","┤"))
    for i, r in enumerate(rows):
        style_str  = f"{r['style_count']} — " + ", ".join(r["styles"])
        style_lines = textwrap.wrap(style_str, W_STYLE)
        voice_lines = textwrap.wrap(r["name"], W_VOICE) or [""]
        n_lines = max(len(style_lines), len(voice_lines))
        for li in range(n_lines):
            s_part = style_lines[li] if li < len(style_lines) else ""
            v_part = voice_lines[li] if li < len(voice_lines) else ""
            n_part = str(i+1) if li == 0 else ""
            print(f"  │ {pad(n_part,W_NUM)} │ {pad(v_part,W_VOICE)} │ {pad(s_part,W_STYLE)} │")
        if i < len(rows)-1:
            print("  " + hline("├","┼","┤"))
    print("  " + hline("└","┴","┘"))
    print()

sections = [
    ("TOP 5 — en-US Female", lambda r: r["locale_group"]=="en" and "US" in r["locale"] and r["gender"]=="Female"),
    ("TOP 5 — en-US Male",   lambda r: r["locale_group"]=="en" and "US" in r["locale"] and r["gender"]=="Male"),
    ("TOP 5 — zh Female",    lambda r: r["locale_group"]=="zh" and r["gender"]=="Female"),
    ("TOP 5 — zh Male",      lambda r: r["locale_group"]=="zh" and r["gender"]=="Male"),
]

for label, fn in sections:
    render_table(label, [r for r in results if fn(r)][:5])
```

Print the output exactly as produced by the script — no reformatting.

---

## Step 2 — Offer to save

After printing, ask the user:
> "Save to a file? (e.g. /tmp/top_voices.txt or press Enter to skip)"
