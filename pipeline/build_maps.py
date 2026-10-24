#!/usr/bin/env python3
"""
ICON-D2 (DWD, ~2,2 km) -> georeferenzierte PNG-Karten fuer die Wetter-App.

Ablauf
  1. Neuesten, vollstaendigen ICON-D2-Lauf auf opendata.dwd.de suchen
  2. GRIB2-Dateien (regular-lat-lon) fuer alle Parameter/Stunden laden
  3. Pro Stunde ein Bild rendern (Web-Mercator, glatt interpoliert, Paletten-PNG mit Transparenz)
  4. manifest.json schreiben (Bounds, Farbskalen, Frames)

Aufruf:   python pipeline/build_maps.py --out site/maps
Optional: --steps 48 --scale 2.0 --workers 8
Benoetigt: numpy, Pillow, eccodes (siehe requirements.txt)
"""
import argparse, bz2, json, math, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import eccodes

BASE = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"
RUN_HOURS = ["00", "03", "06", "09", "12", "15", "18", "21"]
MIN_STEPS = 27          # ein Lauf gilt als "fertig", wenn alle Parameter mind. so viele Stunden haben
UA = {"User-Agent": "gewitterblick-map-builder/1.0"}

# Farbstops: (Anteil 0..1 zwischen min und max, "#RRGGBB" oder "#RRGGBBAA")
# Palette ist an die Farben der App angelehnt.
PARAMS = {
    "t2m": dict(
        var="t_2m", title="Temperatur (2 m)", unit="°C", min=-15, max=40, decimals=0,
        conv=lambda a: a - 273.15,
        stops=[(0.00, "#B18CFF"), (0.20, "#6E9CFF"), (0.40, "#2DD4BF"), (0.55, "#3ECF8E"),
               (0.68, "#F0C93E"), (0.82, "#FF9B45"), (1.00, "#FF5A5A")]),
    "td2m": dict(
        var="td_2m", title="Taupunkt (2 m)", unit="°C", min=-10, max=25, decimals=0,
        conv=lambda a: a - 273.15,
        stops=[(0.00, "#243B6B"), (0.35, "#2DD4BF"), (0.65, "#3ECF8E"), (0.85, "#F0C93E"), (1.00, "#FF9B45")]),
    "precip": dict(
        var="tot_prec", title="Niederschlag (stündlich)", unit="mm", min=0, max=20, decimals=1,
        conv=None, hourly_diff=True, transparent_below=0.1,
        stops=[(0.00, "#2DD4BF99"), (0.05, "#2DD4BFCC"), (0.20, "#6E9CFF"), (0.45, "#B18CFF"),
               (0.70, "#FF7FB0"), (1.00, "#FF5A5A")]),
    "gust": dict(
        var="vmax_10m", title="Windböen (10 m)", unit="km/h", min=0, max=130, decimals=0,
        conv=lambda a: a * 3.6, transparent_below=15,
        stops=[(0.00, "#3ECF8E55"), (0.25, "#3ECF8E"), (0.45, "#F0C93E"), (0.65, "#FF9B45"),
               (0.85, "#FF5A5A"), (1.00, "#B18CFF")]),
    "cape": dict(
        var="cape_ml", title="CAPE (Gewitterenergie)", unit="J/kg", min=0, max=3000, decimals=0,
        conv=None, transparent_below=50,
        stops=[(0.00, "#3ECF8E88"), (0.15, "#3ECF8E"), (0.40, "#F0C93E"), (0.70, "#FF9B45"), (1.00, "#FF5A5A")]),
    "cloud": dict(
        var="clct", title="Bewölkung", unit="%", min=0, max=100, decimals=0,
        conv=None, transparent_below=8,
        stops=[(0.00, "#DCE6FF00"), (0.30, "#DCE6FF55"), (1.00, "#F4F7FFD0")]),
}


def http_get(url, retries=3):
    last = None
    for i in range(retries):
        try:
            with urlopen(Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET fehlgeschlagen: {url}: {last}")


def list_files(hh, var):
    """Dateien eines Verzeichnisses -> {(run, step): url}. Namen werden per Regex gelesen, nicht geraten."""
    url = f"{BASE}/{hh}/{var}/"
    html = http_get(url).decode("utf-8", "ignore")
    out = {}
    pat = re.compile(r'href="([^"]*regular-lat-lon_single-level_(\d{10})_(\d{3})_[^"]*?_' + re.escape(var) + r'\.grib2\.bz2)"')
    for m in pat.finditer(html):
        out[(m.group(2), int(m.group(3)))] = url + m.group(1).split("/")[-1]
    return out


def find_latest_run():
    cands = []
    for hh in RUN_HOURS:
        try:
            files = {k: list_files(hh, p["var"]) for k, p in PARAMS.items()}
        except Exception as e:
            print(f"  {hh} UTC: uebersprungen ({e})", file=sys.stderr)
            continue
        runs = set(r for f in files.values() for (r, _) in f)
        for run in runs:
            steps = {k: sorted(s for (r, s) in f if r == run) for k, f in files.items()}
            n = min(len(v) for v in steps.values())
            print(f"  Lauf {run}: {n} Stunden (min. ueber alle Parameter)")
            if n >= MIN_STEPS:
                cands.append((run, files))
    if not cands:
        raise SystemExit("Kein vollstaendiger ICON-D2-Lauf gefunden.")
    cands.sort(key=lambda c: c[0])
    return cands[-1]


def read_grib(raw_bz2):
    raw = bz2.decompress(raw_bz2)
    gid = eccodes.codes_new_from_message(raw)
    try:
        eccodes.codes_set(gid, "missingValue", 1e20)
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        la1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        la2 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
        lo1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        lo2 = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")
        vals = eccodes.codes_get_values(gid).astype(np.float64).reshape(nj, ni)
    finally:
        eccodes.codes_release(gid)
    vals[vals > 1e19] = np.nan
    if lo1 > 180: lo1 -= 360
    if lo2 > 180: lo2 -= 360
    if la1 > la2:                      # Norden zuerst -> auf Sueden zuerst drehen
        vals = vals[::-1]
        la1, la2 = la2, la1
    return vals, (la1, la2, lo1, lo2)


def mercator_y(lat_deg):
    return np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))


def inv_mercator_y(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


def reproject(vals, extent, scale):
    """Regulaeres Lat/Lon-Gitter -> Web-Mercator-Raster (bilinear, `scale`x hochskaliert)."""
    la1, la2, lo1, lo2 = extent
    nj, ni = vals.shape
    w = int(round(ni * scale))
    y0, y1 = mercator_y(la1), mercator_y(la2)
    h = int(round(w * (y1 - y0) / np.radians(lo2 - lo1)))
    ys = np.linspace(y1, y0, h)                       # Zeile 0 = Norden
    lat = inv_mercator_y(ys)
    fr = (lat - la1) / (la2 - la1) * (nj - 1)         # gebrochene Quellzeile
    fc = np.linspace(0, ni - 1, w)                    # gebrochene Quellspalte
    r0 = np.clip(np.floor(fr).astype(int), 0, nj - 2); tr = (fr - r0)[:, None]
    c0 = np.clip(np.floor(fc).astype(int), 0, ni - 2); tc = (fc - c0)[None, :]
    a = vals[r0][:, c0]; b = vals[r0][:, c0 + 1]
    c = vals[r0 + 1][:, c0]; d = vals[r0 + 1][:, c0 + 1]
    return (a * (1 - tc) + b * tc) * (1 - tr) + (c * (1 - tc) + d * tc) * tr


def hex_rgba(s):
    s = s.lstrip("#")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    a = int(s[6:8], 16) if len(s) >= 8 else 255
    return r, g, b, a


def build_palette(cfg):
    """256-Eintraege: Index 0 = transparent, 1..255 = Farbverlauf ueber min..max."""
    stops = [(p, hex_rgba(c)) for p, c in cfg["stops"]]
    pal = np.zeros((256, 4), dtype=np.uint8)
    for i in range(1, 256):
        t = (i - 1) / 254
        for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
            if p0 <= t <= p1:
                f = 0 if p1 == p0 else (t - p0) / (p1 - p0)
                pal[i] = [round(c0[k] + (c1[k] - c0[k]) * f) for k in range(4)]
                break
    return pal


def render(grid, cfg, pal, path):
    lo, hi = cfg["min"], cfg["max"]
    idx = np.clip((grid - lo) / (hi - lo), 0, 1) * 254 + 1
    idx = np.rint(idx).astype(np.uint8)
    hide = np.isnan(grid)
    if "transparent_below" in cfg:
        hide |= grid < cfg["transparent_below"]
    idx[hide] = 0
    im = Image.fromarray(idx, mode="P")
    im.putpalette(pal[:, :3].reshape(-1).tolist())
    alpha = bytes(pal[:, 3].tolist()); alpha = b"\x00" + alpha[1:]
    im.save(path, optimize=True, transparency=alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/maps")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("Suche neuesten ICON-D2-Lauf ...")
    run, files = find_latest_run()
    run_dt = datetime.strptime(run, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    print("Verwende Lauf", run)

    manifest = {"model": "ICON-D2", "resolution_km": 2.2,
                "run": run_dt.isoformat().replace("+00:00", "Z"),
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "bounds": None, "params": {}}

    for key, cfg in PARAMS.items():
        urls = {s: u for (r, s), u in files[key].items() if r == run and s <= args.steps}
        steps = sorted(urls)
        print(f"[{key}] lade {len(steps)} Dateien ...")
        with ThreadPoolExecutor(args.workers) as ex:
            raws = dict(zip(steps, ex.map(lambda s: http_get(urls[s]), steps)))
        (out / key).mkdir(exist_ok=True)
        pal = build_palette(cfg)
        prev = None
        frames = []
        for s in steps:
            vals, extent = read_grib(raws[s])
            if manifest["bounds"] is None:
                la1, la2, lo1, lo2 = extent
                manifest["bounds"] = [[la1, lo1], [la2, lo2]]
            cur = vals
            if cfg.get("hourly_diff"):
                if s == 0:
                    prev = cur; continue
                vals = np.maximum(cur - (prev if prev is not None else 0), 0)
                prev = cur
            elif cfg.get("conv"):
                vals = cfg["conv"](vals)
            if s == 0 and key in ("gust",):
                continue            # Boeen sind bei Stunde 0 nicht sinnvoll
            grid = reproject(vals, extent, args.scale)
            fn = f"{key}/{s:03d}.png"
            render(grid, cfg, pal, out / fn)
            valid = run_dt + timedelta(hours=s)
            frames.append({"step": s, "valid": valid.isoformat().replace("+00:00", "Z"), "file": fn})
        manifest["params"][key] = {
            "title": cfg["title"], "unit": cfg["unit"], "min": cfg["min"], "max": cfg["max"],
            "decimals": cfg["decimals"], "stops": cfg["stops"], "frames": frames,
        }
        del raws

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
    print("Fertig:", out / "manifest.json")


if __name__ == "__main__":
    main()
