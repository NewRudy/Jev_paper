"""Build the human-baseline form for the ambiguity typology.

Why: every "ideal 0.5/0.5" in this study is currently our own stipulation. A
reviewer's first attack is "humans do not actually disagree on your ambiguous
items". This emits a self-contained HTML form (no build step, no server) that
randomises item order per participant and exports a CSV-shaped answer block.

    python3 gen/make_human_form.py --n 60 --out results/human_form.html

Five annotators, ~15 minutes each. The output reports the empirical choice
distribution per type, which decides whether A1 survives as an ambiguity type
at all (see RESEARCH_PLAN_V2: A4 is the strong one; A1 may be a convention
task rather than a genuinely ambiguous one).
"""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

TYPES = [("amb_a1_stated", "spec stated", "Direction, with a tie-break rule"),
         ("amb_a1_unstated", "spec unstated", "Direction, no tie-break rule"),
         ("amb_a4", "membership", "Subnet relationship"),
         ("amb_a5", "conflict", "Conflicting log entries")]

# The annotator is NOT asked to guess an answer. The question is whether the
# input determines one at all -- a judgement a domain practitioner makes
# routinely, and the one our claim is about.
DETERMINABILITY = ["yes, it is fully determined",
                   "no, the input is under-specified",
                   "cannot tell from this input"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="items per ambiguity type")
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--out", default="results/human_form.html")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    items = []
    for tag, kind, blurb in TYPES:
        p = Path("data") / f"{tag}.jsonl"
        if not p.exists():
            print("missing", p)
            continue
        rows = [json.loads(l) for l in open(p)]
        rng.shuffle(rows)
        for i, r in enumerate(rows[:args.n]):
            q = r["questions"]["q1"]
            items.append({
                "id": f"{tag}-{i:03d}",
                "type": tag, "kind": kind, "blurb": blurb,
                "state": r["state"],
                "question": q["instructions"],
                "options": list(q["criteria"].keys()),
                "key": q["label"],
            })
    rng.shuffle(items)

    blocks = []
    for it in items:
        opts = "".join(
            f'<label class="opt"><input type="radio" name="{it["id"]}" '
            f'value="{html.escape(o)}"> <span>{html.escape(o)}</span></label>'
            for o in DETERMINABILITY)
        blocks.append(f"""
    <fieldset data-type="{it['type']}">
      <legend><span class="tid">{html.escape(it['id'])}</span>
        <span class="k">{html.escape(it['kind'])}</span></legend>
      <pre class="state">{html.escape(it['state'])}</pre>
      <p class="q">{html.escape(it['question'])}</p>
      <p class="ask"><strong>From this input alone, can the answer be
        uniquely determined?</strong></p>
      <div class="opts">{opts}</div>
    </fieldset>""")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Ambiguity annotation form</title>
<style>
 body {{ font: 15px/1.55 -apple-system, "Segoe UI", system-ui, sans-serif;
        max-width: 760px; margin: 0 auto; padding: 24px 20px 80px; color: #0b0b0b; }}
 h1 {{ font-size: 20px; margin: 0 0 4px; }}
 .lede {{ color: #52514e; font-size: 14px; margin: 0 0 20px; }}
 fieldset {{ border: 1px solid #e4e4e0; border-radius: 6px; padding: 14px 16px;
             margin: 0 0 18px; }}
 legend {{ font-size: 12px; color: #52514e; padding: 0 6px; }}
 .tid {{ font-family: ui-monospace, monospace; }}
 .k {{ margin-left: 8px; padding: 1px 7px; border-radius: 10px;
        background: #f0efeb; font-size: 11px; }}
 pre.state {{ white-space: pre-wrap; background: #f7f7f4; padding: 10px 12px;
              border-radius: 4px; font-size: 13px; margin: 8px 0; }}
 p.q {{ font-weight: 600; margin: 8px 0; }}
 .opts {{ display: flex; flex-wrap: wrap; gap: 6px 20px; }}
 .opt {{ display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }}
 p.ask {{ font-size: 14px; margin: 10px 0 6px; }}
 .bar {{ position: sticky; bottom: 0; background: #fff; border-top: 1px solid #e4e4e0;
         padding: 12px 0; display: flex; gap: 12px; align-items: center; }}
 button {{ font: inherit; padding: 8px 18px; border-radius: 6px;
           border: 1px solid #2a78d6; background: #2a78d6; color: #fff; cursor: pointer; }}
 button.ghost {{ background: #fff; color: #2a78d6; }}
 #status {{ font-size: 13px; color: #52514e; }}
</style></head><body>
<h1>Ambiguity annotation form</h1>
<p class="lede">{len(items)} items, presented in random order. For each one,
answer a single question: <strong>does this input determine a unique answer,
or is it under-specified?</strong> You are not being asked to guess an answer.
If two readings are both defensible, the input is under-specified. Work from
the text only; do not look anything up.</p>
<form id="f">{''.join(blocks)}
  <div class="bar">
    <button type="button" onclick="exportCSV()">Export answers (CSV)</button>
    <button type="button" class="ghost" onclick="window.print()">Print / PDF</button>
    <span id="status"></span>
  </div>
</form>
<script>
const META = {json.dumps(items, ensure_ascii=False)};
function exportCSV() {{
  const rows = [["item_id", "type", "determinability"]];
  let done = 0;
  for (const it of META) {{
    const a = document.querySelector('input[name="' + it.id + '"]:checked');
    if (a) done++;
    rows.push([it.id, it.type, a ? a.value : ""]);
  }}
  const csv = rows.map(r => r.map(x => '"' + String(x).replace(/"/g, '""') + '"').join(",")).join("\\n");
  const blob = new Blob([csv], {{type: "text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ambiguity_answers.csv";
  a.click();
  document.getElementById("status").textContent =
    done + " / " + META.length + " answered — file downloaded";
}}
</script></body></html>"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    key = out.with_suffix(".key.json")
    key.write_text(json.dumps({i["id"]: i["key"] for i in items}, indent=2))
    print(f"wrote {out} ({len(items)} items) and {key}")


if __name__ == "__main__":
    main()
