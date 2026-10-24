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

# Farbstops: (Anteil 0..1 zwischen min und max, "#RRGGBB" oder "#RRGGBBAA"); gamma<1 streckt kleine Werte.
# Parameter, die der DWD nicht (mehr) liefert, werden automatisch uebersprungen.
K = lambda a: a - 273.15
TEMP = [(0,"#8A2BE2"),(.12,"#4B3FD8"),(.25,"#2E7BFF"),(.38,"#27C6E8"),(.5,"#3ECF8E"),(.6,"#B6E34A"),(.7,"#FFE14A"),(.8,"#FF9B45"),(.9,"#FF4D4D"),(1,"#C2185B")]
PARAMS = {
 "t2m":   dict(vars=["t_2m"], title="Temperatur (2 m)", unit="°C", min=-20, max=40, decimals=0, conv=K, stops=TEMP),
 "td2m":  dict(vars=["td_2m"], title="Taupunkt (2 m)", unit="°C", min=-20, max=25, decimals=0, conv=K, stops=TEMP),
 "rh":    dict(vars=["relhum_2m"], title="Luftfeuchte (2 m)", unit="%", min=20, max=100, decimals=0,
               stops=[(0,"#E8B04A"),(.4,"#F4E9A0"),(.65,"#8EDBA8"),(.85,"#3AA8D8"),(1,"#2B3FBF")]),
 "precip":dict(vars=["tot_prec"], title="Niederschlag (stündlich)", unit="mm", min=0, max=30, decimals=1, gamma=.5,
               hourly_diff=True, transparent_below=0.1,
               stops=[(0,"#A8E6A0B0"),(.1,"#5CCB5C"),(.25,"#2EA84A"),(.4,"#F2E84A"),(.55,"#FFA13D"),(.75,"#FF4D4D"),(.9,"#C21BC2"),(1,"#6A1B9A")]),
 "ptotal":dict(vars=["tot_prec"], title="Niederschlag (Summe seit Start)", unit="mm", min=0, max=100, decimals=0, gamma=.5,
               transparent_below=0.5,
               stops=[(0,"#A8E6A0B0"),(.1,"#5CCB5C"),(.25,"#2EA84A"),(.4,"#F2E84A"),(.55,"#FFA13D"),(.75,"#FF4D4D"),(.9,"#C21BC2"),(1,"#6A1B9A")]),
 "wind":  dict(vars=["u_10m","v_10m"], title="Wind (10 m, Mittel)", unit="km/h", min=0, max=100, decimals=0,
               comb=lambda u,v: np.hypot(u,v)*3.6, transparent_below=5,
               stops=[(0,"#7FDBFF66"),(.15,"#3ECF8E"),(.35,"#B6E34A"),(.5,"#FFE14A"),(.7,"#FF9B45"),(.85,"#FF4D4D"),(1,"#B18CFF")]),
 "gust":  dict(vars=["vmax_10m"], title="Windböen (10 m)", unit="km/h", min=0, max=150, decimals=0,
               conv=lambda a: a*3.6, transparent_below=20, skip_step0=True,
               stops=[(0,"#7FDBFF66"),(.2,"#3ECF8E"),(.35,"#B6E34A"),(.5,"#FFE14A"),(.65,"#FF9B45"),(.8,"#FF4D4D"),(.9,"#C21BC2"),(1,"#6A1B9A")]),
 "cape":  dict(vars=["cape_ml"], title="CAPE (Gewitterenergie)", unit="J/kg", min=0, max=3000, decimals=0, gamma=.6,
               transparent_below=50,
               stops=[(0,"#3ECF8E88"),(.12,"#3ECF8E"),(.3,"#F0C93E"),(.55,"#FF9B45"),(.8,"#FF4D4D"),(1,"#C21BC2")]),
 "cin":   dict(vars=["cin_ml"], title="CIN (Gewitterbremse)", unit="J/kg", min=0, max=300, decimals=0,
               conv=np.abs, transparent_below=5,
               stops=[(0,"#6E9CFF88"),(.3,"#6E9CFF"),(.6,"#B18CFF"),(1,"#6A1B9A")]),
 "cloud": dict(vars=["clct"], title="Bewölkung", unit="%", min=0, max=100, decimals=0, transparent_below=8,
               stops=[(0,"#DCE6FF00"),(.3,"#DCE6FF55"),(1,"#F4F7FFD0")]),
 "pmsl":  dict(vars=["pmsl"], title="Luftdruck (auf Meereshöhe)", unit="hPa", min=980, max=1040, decimals=0,
               conv=lambda a: a/100,
               stops=[(0,"#6A1B9A"),(.25,"#2E7BFF"),(.5,"#3ECF8E"),(.75,"#FFE14A"),(1,"#FF4D4D")]),
 "freezing": dict(vars=["hzerocl"], title="Nullgradgrenze", unit="m", min=0, max=4500, decimals=0,
               stops=[(0,"#6E9CFF"),(.3,"#27C6E8"),(.55,"#3ECF8E"),(.8,"#FFE14A"),(1,"#FF4D4D")]),
 "snowlmt": dict(vars=["snowlmt"], title="Schneefallgrenze", unit="m", min=0, max=3500, decimals=0,
               stops=[(0,"#FFFFFF"),(.25,"#B3D4FF"),(.55,"#6E9CFF"),(.8,"#B18CFF"),(1,"#6A1B9A")]),
 "snowdepth": dict(vars=["h_snow"], title="Schneehöhe", unit="cm", min=0, max=60, decimals=0, gamma=.6,
               conv=lambda a: a*100, transparent_below=1,
               stops=[(0,"#DCE6FFAA"),(.25,"#8FB8FF"),(.6,"#6E9CFF"),(1,"#FFFFFF")]),
}
ALLVARS = sorted({v for p in PARAMS.values() for v in p["vars"]})


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
    """Neuester Lauf, bei dem die Kernparameter komplett sind. Fehlende Zusatzparameter werden uebersprungen."""
    CORE = ["t_2m", "tot_prec", "cape_ml"]
    cands = []
    for hh in RUN_HOURS:
        files = {}
        for v in ALLVARS:
            try:
                files[v] = list_files(hh, v)
            except Exception as e:
                print(f"  {hh} UTC {v}: nicht verfuegbar", file=sys.stderr)
        for run in set(r for f in files.values() for (r, _) in f):
            n = lambda v: len([1 for (r, _) in files.get(v, {}) if r == run])
            print(f"  Lauf {run}: t_2m {n('t_2m')} Stunden")
            if all(n(v) >= MIN_STEPS for v in CORE):
                cands.append((run, {v: {k: u for k, u in f.items() if k[0] == run} for v, f in files.items()}))
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
    idx = np.clip((grid - lo) / (hi - lo), 0, 1) ** cfg.get("gamma", 1) * 254 + 1
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
    cache = {}                                   # (var, step) -> Gitter; mehrere Karten nutzen dieselbe Variable

    def load(var, step):
        if (var, step) not in cache:
            vals, ext = read_grib(http_get(files[var][(run, step)]))
            cache[(var, step)] = (vals, ext)
        return cache[(var, step)]

    for key, cfg in PARAMS.items():
        if any(v not in files or not any(r == run for (r, _) in files[v]) for v in cfg["vars"]):
            print(f"[{key}] uebersprungen (nicht im DWD-Angebot)")
            continue
        steps = sorted(s for (r, s) in files[cfg["vars"][0]] if s <= args.steps)
        steps = [s for s in steps if all((run, s) in files[v] for v in cfg["vars"])]
        print(f"[{key}] {len(steps)} Stunden")
        with ThreadPoolExecutor(args.workers) as ex:   # Downloads parallel vorab
            list(ex.map(lambda sv: load(sv[1], sv[0]) if (sv[1], sv[0]) not in cache else None,
                        [(s, v) for s in steps for v in cfg["vars"]]))
        (out / key).mkdir(exist_ok=True)
        pal = build_palette(cfg); prev = None; frames = []
        for s in steps:
            arrs = [load(v, s) for v in cfg["vars"]]
            extent = arrs[0][1]
            if manifest["bounds"] is None:
                manifest["bounds"] = [[extent[0], extent[2]], [extent[1], extent[3]]]
            if "comb" in cfg:
                vals = cfg["comb"](*[a[0] for a in arrs])
            else:
                vals = arrs[0][0]
                if cfg.get("hourly_diff"):
                    cur = vals
                    if s == 0 or prev is None:
                        prev = cur; continue
                    vals = np.maximum(cur - prev, 0); prev = cur
                if cfg.get("conv"):
                    vals = cfg["conv"](vals)
            if s == 0 and cfg.get("skip_step0"):
                continue
            render(reproject(vals, extent, args.scale), cfg, pal, out / key / f"{s:03d}.png")
            valid = run_dt + timedelta(hours=s)
            frames.append({"step": s, "valid": valid.isoformat().replace("+00:00", "Z"), "file": f"{key}/{s:03d}.png"})
        manifest["params"][key] = {"title": cfg["title"], "unit": cfg["unit"], "min": cfg["min"], "max": cfg["max"],
                                   "decimals": cfg["decimals"], "gamma": cfg.get("gamma", 1),
                                   "stops": cfg["stops"], "frames": frames}
        if cfg["vars"][0] != "tot_prec":
            for s in steps: [cache.pop((v, s), None) for v in cfg["vars"]]   # Speicher freigeben

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
    print("Fertig:", out / "manifest.json")


if __name__ == "__main__":
    main()
