"""
visualizer.py - HTML visual log for the adaptive-reasoning loop.

A text .log can't show images. This writes a self-contained .html file where each
frame shows: the actual CROPPED sign image (embedded), what the detectors saw, the
read + confidence, the VLM chain-of-thought reasoning, and the final action —
color-coded. Open it in a browser; it rewrites after every frame so you can
refresh to watch progress live.

Usage (wired into the loop): the loop calls viz.add_frame(...) per frame and
viz.flush() writes the HTML. Self-contained (images base64-embedded), so you can
scp the single .html off the Jetson and open it anywhere.
"""

import base64
import html
import io
from datetime import datetime
from pathlib import Path


# action -> color for quick visual scanning
ACTION_COLORS = {
    "turn_left": "#3b82f6", "turn_right": "#8b5cf6", "forward": "#10b981",
    "go_straight": "#10b981", "stop": "#ef4444", "reroute": "#f59e0b",
    "continue": "#6b7280", "approach": "#06b6d4",
}


class HTMLVisualizer:
    def __init__(self, out_path: str, goal: str = ""):
        self.out_path = Path(out_path)
        self.goal = goal
        self.frames = []        # list of dicts, one per processed frame
        self.started = datetime.now()

    def add_frame(self, idx, ts, sign_dets, hazard_dets, chosen_cls, chosen_conf,
                  crop_pil=None, crop_size=None, crop_frac=None,
                  read_conf=None, parsed=None, reasoning=None, action=None,
                  branch=None):
        """Record one frame. crop_pil is a PIL image (the actual crop) or None."""
        crop_b64 = ""
        if crop_pil is not None:
            try:
                buf = io.BytesIO()
                crop_pil.save(buf, format="JPEG", quality=85)
                crop_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception:
                crop_b64 = ""
        self.frames.append(dict(
            idx=idx, ts=ts, sign_dets=sign_dets or [], hazard_dets=hazard_dets or [],
            chosen_cls=chosen_cls, chosen_conf=chosen_conf, crop_b64=crop_b64,
            crop_size=crop_size, crop_frac=crop_frac, read_conf=read_conf,
            parsed=parsed, reasoning=reasoning, action=action, branch=branch,
        ))

    def flush(self):
        """Write the full HTML file (call after each frame so it stays current)."""
        self.out_path.write_text(self._render(), encoding="utf-8")

    # ---------- rendering ----------
    def _render(self) -> str:
        cards = "\n".join(self._card(f) for f in reversed(self.frames))  # newest first
        n = len(self.frames)
        n_signs = sum(1 for f in self.frames if f["branch"] == "sign")
        n_haz = sum(1 for f in self.frames if f["branch"] == "hazard")
        return _PAGE.format(
            goal=html.escape(self.goal), n=n, n_signs=n_signs, n_haz=n_haz,
            started=self.started.strftime("%Y-%m-%d %H:%M:%S"),
            updated=datetime.now().strftime("%H:%M:%S"), cards=cards)

    def _card(self, f) -> str:
        action = f.get("action") or "—"
        color = ACTION_COLORS.get(action, "#6b7280")
        # crop image
        if f["crop_b64"]:
            sz = f.get("crop_size") or ("?", "?")
            frac = f.get("crop_frac")
            frac_s = f" · {frac*100:.1f}% of frame" if frac else ""
            small = ""
            try:
                if f.get("crop_size") and (f["crop_size"][0] < 80 or f["crop_size"][1] < 80):
                    small = '<div class="warn">⚠ crop small — may be too far to read</div>'
            except Exception:
                pass
            img_html = (f'<img class="crop" src="data:image/jpeg;base64,{f["crop_b64"]}"/>'
                        f'<div class="cropmeta">{sz[0]}×{sz[1]}px{frac_s}</div>{small}')
        else:
            img_html = '<div class="nocrop">no crop<br>(no sign branch)</div>'

        # detectors
        det_rows = []
        for d in f["sign_dets"][:3]:
            det_rows.append(f'<span class="tag sign">sign {html.escape(str(d.get("class_name","")))} '
                            f'{d.get("confidence",0):.2f}</span>')
        for d in f["hazard_dets"][:3]:
            det_rows.append(f'<span class="tag haz">hazard {html.escape(str(d.get("label","")))} '
                            f'{d.get("confidence",0):.2f}</span>')
        if not det_rows:
            det_rows.append('<span class="tag none">nothing detected</span>')
        det_html = " ".join(det_rows)

        # read
        read_html = ""
        if f.get("read_conf") is not None:
            parsed = f.get("parsed") or {}
            parsed_str = ", ".join(f"{html.escape(str(k))}→{html.escape(str(v))}"
                                   for k, v in parsed.items()) or "(empty)"
            conf = f["read_conf"]
            conf_cls = "good" if conf >= 0.7 else "low"
            read_html = (f'<div class="read"><b>read</b> '
                         f'<span class="conf {conf_cls}">conf {conf:.2f}</span> '
                         f'<span class="parsed">{parsed_str}</span></div>')

        # reasoning
        reasoning_html = ""
        if f.get("reasoning"):
            reasoning_html = f'<div class="reasoning">{html.escape(f["reasoning"])}</div>'

        branch = f.get("branch") or "—"
        return f'''
<div class="card">
  <div class="left">{img_html}</div>
  <div class="right">
    <div class="head">
      <span class="frame">frame {f["idx"]}</span>
      <span class="ts">{html.escape(str(f.get("ts","")))}</span>
      <span class="branch b-{branch}">{branch}</span>
      <span class="action" style="background:{color}">{html.escape(action)}</span>
    </div>
    <div class="dets">{det_html}</div>
    {read_html}
    {reasoning_html}
  </div>
</div>'''


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>SignNav live reasoning</title>
<meta http-equiv="refresh" content="5">
<style>
  body {{ background:#0f1115; color:#e5e7eb; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
         margin:0; padding:24px; }}
  h1 {{ font-size:18px; margin:0 0 4px; }}
  .sub {{ color:#9ca3af; font-size:13px; margin-bottom:18px; }}
  .stats {{ display:flex; gap:18px; margin-bottom:20px; }}
  .stat {{ background:#1a1d24; border:1px solid #2a2e38; border-radius:10px; padding:10px 16px; }}
  .stat b {{ font-size:22px; }} .stat span {{ color:#9ca3af; font-size:12px; display:block; }}
  .card {{ display:flex; gap:18px; background:#1a1d24; border:1px solid #2a2e38;
          border-radius:12px; padding:16px; margin-bottom:14px; }}
  .left {{ flex:0 0 220px; }}
  .crop {{ width:220px; border-radius:8px; border:1px solid #3a3e48; display:block; }}
  .cropmeta {{ color:#9ca3af; font-size:11px; margin-top:6px; }}
  .nocrop {{ width:220px; height:130px; display:flex; align-items:center; justify-content:center;
            background:#13151a; border:1px dashed #3a3e48; border-radius:8px; color:#6b7280;
            text-align:center; font-size:12px; }}
  .warn {{ color:#f59e0b; font-size:11px; margin-top:4px; }}
  .right {{ flex:1; min-width:0; }}
  .head {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; flex-wrap:wrap; }}
  .frame {{ font-weight:700; }} .ts {{ color:#9ca3af; font-size:12px; }}
  .branch {{ font-size:11px; padding:2px 8px; border-radius:20px; border:1px solid #3a3e48; }}
  .b-sign {{ color:#60a5fa; }} .b-hazard {{ color:#f87171; }} .b-none {{ color:#6b7280; }}
  .action {{ margin-left:auto; color:#fff; font-weight:700; font-size:12px;
            padding:4px 12px; border-radius:20px; }}
  .dets {{ margin-bottom:8px; }}
  .tag {{ display:inline-block; font-size:11px; padding:2px 8px; border-radius:6px;
         margin:0 4px 4px 0; }}
  .tag.sign {{ background:#1e3a5f; color:#93c5fd; }}
  .tag.haz {{ background:#5f1e1e; color:#fca5a5; }}
  .tag.none {{ background:#26292f; color:#9ca3af; }}
  .read {{ font-size:13px; margin:8px 0; }}
  .conf {{ padding:1px 7px; border-radius:5px; font-weight:600; }}
  .conf.good {{ background:#14361f; color:#86efac; }}
  .conf.low {{ background:#3a2a14; color:#fcd34d; }}
  .parsed {{ color:#cbd5e1; }}
  .reasoning {{ background:#13151a; border-left:3px solid #3b82f6; border-radius:6px;
               padding:10px 12px; margin-top:8px; font-size:12.5px; line-height:1.5;
               white-space:pre-wrap; color:#cbd5e1; font-family:ui-monospace,monospace; }}
</style></head>
<body>
  <h1>SignNav — live reasoning</h1>
  <div class="sub">goal: <b>{goal}</b> · started {started} · updated {updated}
      · auto-refreshes every 5s</div>
  <div class="stats">
    <div class="stat"><b>{n}</b><span>frames</span></div>
    <div class="stat"><b>{n_signs}</b><span>sign reasoning</span></div>
    <div class="stat"><b>{n_haz}</b><span>hazard reasoning</span></div>
  </div>
  {cards}
</body></html>"""