"""S1 data-availability probe for the siting-compliance study (RESEARCH_PLAN_V4 §2).

Answers the first kill-or-continue question: can we actually get the layers a
real county screening needs, with no credentials and no institutional access?

Everything here is public, keyless, and was verified reachable on 2026-09-29.
Tile naming was learned by listing the S3 buckets, not guessed -- both schemes
below are non-obvious:

  ESA WorldCover 2021 : 3x3-degree tiles, N{lat0:02d}E{lon0:03d}, lat0 = floor(lat/3)*3
  Copernicus DEM GLO-30: 1x1-degree tiles, N{lat:02d}_00_E{lon:03d}_00_DEM

Run:  python3 gen/probe_siting_data.py
"""
from __future__ import annotations

import urllib.request

UA = {"User-Agent": "spatial-siting-screener/1.0 (research; contact: research@example.org)"}
WC = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{}"
DEM = "https://copernicus-dem-30m.s3.amazonaws.com/{d}/{d}.tif"

CANDIDATES = [
    ("贵州威宁", 26.66, 104.28, "贵州大型风电基地，院内业务熟悉"),
    ("贵州赫章", 27.12, 104.72, "贵州毕节片区，喀斯特地形可检验坡度约束"),
    ("甘肃瓜州", 40.52, 95.78, "全国最大陆上风电基地，戈壁地类单一是缺点"),
    ("青海德令哈", 37.37, 97.36, "光伏基地，海拔高、地形开阔"),
    ("青海共和", 36.14, 100.50, "塔拉滩光伏园区"),
]


def probe(url: str, label: str, sample_bytes: int = 150_000) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            n = len(r.read(sample_bytes))
            return True, f"{r.status}, first {n // 1024}KB readable"
    except Exception as e:
        return False, str(e)[:60]


def main():
    print(f"{'候选县':<12} {'WorldCover 10m':<34} {'Copernicus DEM 30m':<34} 备注")
    print("-" * 118)
    ok_all = []
    for name, lat, lon, note in CANDIDATES:
        lat0, lon0 = int(lat // 3) * 3, int(lon // 3) * 3
        wc_key = f"ESA_WorldCover_10m_2021_v200_N{lat0:02d}E{lon0:03d}_Map.tif"
        ok1, msg1 = probe(WC.format(wc_key), wc_key)
        d = f"Copernicus_DSM_COG_10_N{int(lat):02d}_00_E{int(lon):03d}_00_DEM"
        ok2, msg2 = probe(DEM.format(d=d), d)
        print(f"{name:<12} {(msg1 if ok1 else 'FAIL ' + msg1):<34} {(msg2 if ok2 else 'FAIL ' + msg2):<34} {note}")
        if ok1 and ok2:
            ok_all.append((name, lat, lon))

    print()
    if ok_all:
        print(f"{len(ok_all)}/{len(CANDIDATES)} 个候选县的两类核心数据均可免鉴权下载：")
        for n, la, lo in ok_all:
            print(f"  - {n} ({la}, {lo})")
    else:
        print("!! 无候选县通过核验")
    print()
    print("仍需人工/院内核验的图层（公开但来源分散，见 RESEARCH_PLAN_V4 §2）：")
    print("  - 生态保护红线 / 永久基本农田：省级公开矢量数据，格式与更新频率因省而异")
    print("  - 电网与道路：OpenStreetMap（Overpass API），需确认该县覆盖质量")
    print("  - 风/光资源：Global Wind Atlas / Global Solar Atlas（网页下载或 API）")
    print("  - 真实约束条款：该县/省公开发布的规划管理文件（需业务人员核对）")


if __name__ == "__main__":
    main()
