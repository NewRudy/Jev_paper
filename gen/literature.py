"""Literature lookup that does not get rate-limited.

Why this exists: the arXiv API is unauthenticated and throttles by IP (three
queries in a minute is enough to trip it) and Semantic Scholar returns 429
without a key. OpenAlex accepts anonymous requests at a usable rate and
indexes BOTH arXiv preprints and published journal articles, with venue,
citation counts and references -- so it is the primary source here. arXiv is
kept only as a fallback and is called with a polite User-Agent and backoff.

    python3 gen/literature.py find "compositional generalization" --since 2026-01-01
    python3 gen/literature.py venue "Applied Intelligence" --search calibration
    python3 gen/literature.py related 2606.18089

Results are cached under .litcache/ so repeated lookups cost nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(".litcache")
CACHE.mkdir(exist_ok=True)
UA = "spatial-systemone-literature/1.0 (mailto:research@example.org)"
MAIL = "mailto=research@example.org"


def _get(url: str, cache_key: str, retries: int = 3) -> dict:
    cf = CACHE / (cache_key + ".json")
    if cf.exists():
        return json.loads(cf.read_text())
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                data = json.loads(r.read())
            cf.write_text(json.dumps(data))
            return data
        except Exception as e:                  # back off, then retry
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"fetch failed after {retries} tries: {last}")


def _oa(params: dict, key: str) -> dict:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params) + "&" + MAIL
    return _get(url, key)


def _fmt(w: dict) -> str:
    loc = w.get("primary_location") or {}
    src = (loc.get("source") or {}).get("display_name") or "?"
    yr = w.get("publication_year") or "?"
    cites = w.get("cited_by_count", 0)
    doi = (w.get("doi") or "")[:60]
    return f"[{yr}] cites={cites:<4} {src[:34]:34s} {w.get('title','')[:88]}"


def cmd_find(args):
    params = {
        "search": args.query,
        "per-page": str(args.limit),
        "sort": "publication_date:desc",
    }
    if args.since:
        params["filter"] = f"from_publication_date:{args.since}"
    d = _oa(params, "find_" + urllib.parse.quote(args.query) + (args.since or ""))
    print(f"OpenAlex: {d.get('meta',{}).get('count','?')} total matches\n")
    for w in d.get("results", []):
        print(" ", _fmt(w))


def cmd_venue(args):
    """Papers in a named journal. OpenAlex source lookup by display name."""
    d = _get("https://api.openalex.org/sources?" + urllib.parse.urlencode(
        {"search": args.venue, "per-page": "5"}) + "&" + MAIL, "src_" + args.venue.replace(" ", "_"))
    srcs = d.get("results", [])
    if not srcs:
        print("source not found:", args.venue)
        return
    for s in srcs[:3]:
        print(f"  source: {s['display_name']}  ({s.get('works_count')} works, {s.get('id')})")
    sid = srcs[0]["id"].rsplit("/", 1)[-1]
    flt = f"primary_location.source.id:{sid}"
    if args.since:
        flt += f",from_publication_date:{args.since}"
    if args.search:
        params = {"filter": flt, "search": args.search, "per-page": str(args.limit),
                  "sort": "publication_date:desc"}
    else:
        params = {"filter": flt, "per-page": str(args.limit), "sort": "cited_by_count:desc"}
    d2 = _oa(params, f"venue_{sid}_{args.search or 'top'}_{args.since or ''}")
    print(f"\n{srcs[0]['display_name']}: {d2.get('meta',{}).get('count','?')} matches\n")
    for w in d2.get("results", []):
        print(" ", _fmt(w))


def cmd_related(args):
    """What a given arXiv/OpenAlex id cites and is cited by."""
    aid = args.id
    key = aid.replace("/", "_")
    d = _get(f"https://api.openalex.org/works/https://arxiv.org/abs/{aid}?{MAIL}", "rel_" + key)
    print("TITLE:", d.get("title"))
    print("YEAR :", d.get("publication_year"), "| cites:", d.get("cited_by_count"))
    loc = d.get("primary_location") or {}
    print("VENUE:", (loc.get("source") or {}).get("display_name"))
    print("\nREFERENCES (most cited first):")
    refs = []
    for r in d.get("referenced_works", []):
        rid = r.rsplit("/", 1)[-1]
        try:
            rd = _get(f"https://api.openalex.org/works/{rid}?{MAIL}", "w_" + rid)
            refs.append((rd.get("cited_by_count", 0), rd.get("title") or "?", rd.get("publication_year")))
        except Exception:
            pass
    for c, t, y in sorted(refs, reverse=True)[:15]:
        print(f"   [{y}] {c:5d}  {t[:92]}")
    print("\nCITED BY:")
    d2 = _get("https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"filter": f"cites:{d['id'].rsplit('/',1)[-1]}", "per-page": "20",
         "sort": "publication_date:desc"}) + "&" + MAIL, "cited_" + key)
    for w in d2.get("results", []):
        print(" ", _fmt(w))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find");   f.add_argument("query"); f.add_argument("--since"); f.add_argument("--limit", type=int, default=12)
    v = sub.add_parser("venue");  v.add_argument("venue"); v.add_argument("--search"); v.add_argument("--since"); v.add_argument("--limit", type=int, default=12)
    r = sub.add_parser("related"); r.add_argument("id")
    a = ap.parse_args()
    {"find": cmd_find, "venue": cmd_venue, "related": cmd_related}[a.cmd](a)


if __name__ == "__main__":
    main()
