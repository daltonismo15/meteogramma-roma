#!/usr/bin/env python3
"""Rome · multi-model meteogram, every morning (GitHub Actions).

Downloads the latest run of ICON-2I, WRF, MOLOCH, BOLAM (MeteoHub, ItaliaMeteo) and GFS (NOAA) at Rome,
draws the meteogram of the next hours, and updates README.md with it. Same code as the notebook.

    python meteogram.py --hours 15 --out meteograms
"""
import matplotlib
matplotlib.use("Agg")                                     # no screen on the server
# The places: name, latitude, longitude, height of the real ground (m), short name for the files
PLACES = {
    "Rome (Piazza Venezia)": dict(lat=41.8960, lon=12.4823, z=20, tag="Rome"),
}
PLACE = dict(name="Rome (Piazza Venezia)", **PLACES["Rome (Piazza Venezia)"])     # changed by the menu

# ================================================================ setup: installs what Colab lacks (about 20 s the first time)
import importlib.util, subprocess, sys
if importlib.util.find_spec("eccodes") is None:
    print("installing eccodes ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "eccodes"], check=False)
    importlib.invalidate_caches()
import os, re, csv, json, time, struct, hashlib, tempfile, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode, urlparse
import numpy as np, requests
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import eccodes                                   # reads the GRIB2 files
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 100})

CACHE = os.path.join(tempfile.gettempdir(), "multimodel_meteogram")
os.makedirs(CACHE, exist_ok=True)
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "multimodel-meteogram (personal use; python-requests)"
HTTP.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=32, max_retries=1))
STATS = {"bytes": 0, "requests": 0}
_lock = threading.Lock()
METEOHUB = os.environ.get("METEOHUB_URL", "https://meteohub.agenziaitaliameteo.it")

# ================================================================ the models
# kind "nwp":    MeteoHub folder /nwp/<folder>/<run YYYYMMDDHH>/<variable>/<one GRIB file with all the forecast hours>
# kind "bundle": MeteoHub open-data bundle, one GRIB file with a few variables at all the forecast hours
MODELS = {
    "ICON-2I": dict(kind="nwp", folder="ICON-2I_SURFACE_PRESSURE_LEVELS", color="#1f77b4", res="2.2 km",
                    who="ItaliaMeteo (ARPAE), Italian implementation of the DWD model ICON",
                    vars={"T_2M": "t2", "TD_2M": "td2", "U_10M": "u10", "V_10M": "v10", "VMAX_10M": "gust", "PMSL": "mslp",
                          "TOT_PREC": "tp", "HSURF": "zs"}),
    "WRF": dict(kind="nwp", folder="WRF", color="#d62728", res="1.5 km",
                who="operational chain of ItaliaMeteo, CIMA Foundation and LaMMA (Tuscany)",
                vars={"2t": "t2", "2d": "td2", "10u": "u10", "10v": "v10", "gust": "gust", "prmsl": "mslp", "tp": "tp",
                      "orog": "zs"}),
    "MOLOCH": dict(kind="bundle", dataset="MOLOCH", color="#2ca02c", res="1.2 km", who="CNR-ISAC (Bologna)"),
    "BOLAM": dict(kind="bundle", dataset="BOLAM", color="#9467bd", res="8 km", who="CNR-ISAC (Bologna)"),
}
GFS_STYLE = dict(color="#222222", res="0.25°, about 25 km", who="NOAA/NCEP (USA), global model")
# GRIB2 (discipline, category, number, type of level, level) -> variable, for the bundles (and as a check for the folders)
GRIB_VARS = {(0, 0, 0, 103, 2): "t2", (0, 0, 6, 103, 2): "td2", (0, 2, 2, 103, 10): "u10", (0, 2, 3, 103, 10): "v10",
             (0, 2, 22, 103, 10): "gust", (0, 2, 22, 1, 0): "gust", (0, 3, 1, 101, 0): "mslp", (0, 1, 8, 1, 0): "tp",
             (0, 1, 52, 1, 0): "tp_rate", (0, 3, 5, 1, 0): "zs", (0, 3, 6, 1, 0): "zs"}


# ================================================================ HTTP: whole small files, or byte ranges of big ones
def http_get(url, start=None, end=None, timeout=60, tries=3):
    """bytes of url (or of the byte range start..end); None if not found. A server that ignores the range is refused:
    the bundles are up to 1 GB, and only a few bytes are needed."""
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    for k in range(tries):
        try:
            with HTTP.get(url, headers=headers, timeout=timeout, stream=True) as r:
                if r.status_code == 404:
                    return None
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 * (k + 1)); continue
                if start is not None and r.status_code != 206:
                    raise RuntimeError(f"{urlparse(url).netloc} does not accept byte ranges (HTTP {r.status_code})")
                r.raise_for_status()
                data = r.content
            with _lock:
                STATS["bytes"] += len(data); STATS["requests"] += 1
            return data
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"{urlparse(url).netloc} cannot be reached (after {tries} tries)")


def listing(url):
    """names in an 'Index of' page of the server"""
    page = http_get(url, timeout=30)
    return [] if page is None else re.findall(r'href="([^"?/][^"]*)"', page.decode("utf-8", "replace"))


def pmap(fun, items, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fun, items))


# ================================================================ GRIB2: the header of one message, read from a byte range
def _sm(b):
    """signed integer in GRIB2 sign-and-magnitude form"""
    v = int.from_bytes(b, "big")
    top = 1 << (8 * len(b) - 1)
    return -(v & (top - 1)) if v & top else v


def parse_header(buf, off):
    """the header of the GRIB2 message that starts at byte 0 of buf (= byte off of the file); None if buf is too short"""
    if len(buf) < 16 or buf[:4] != b"GRIB" or buf[7] != 2:
        raise ValueError(f"no GRIB2 message at byte {off}")
    m = {"off": off, "total": struct.unpack(">Q", buf[8:16])[0], "disc": buf[6]}
    p = 16
    while True:
        if p + 5 > len(buf):
            return None
        n, num = struct.unpack(">I", buf[p:p + 4])[0], buf[p + 4]
        if num == 7:
            m["s7"] = off + p
            m["head"] = bytes(buf[16:p])                          # sections 1 to 6, for eccodes
            m["rel7"] = p
            return m
        if p + n > len(buf):
            return None
        s = buf[p:p + n]
        if num == 1:
            m["ref"] = dt.datetime(struct.unpack(">H", s[12:14])[0], s[14], s[15], s[16], s[17])
        elif num == 3:
            m["sec3"] = bytes(s)
        elif num == 4:
            m["t4"] = struct.unpack(">H", s[7:9])[0]
            m["cat"], m["num"] = s[9], s[10]
            m["lev_type"] = s[22]
            sc, val = s[23], struct.unpack(">I", s[24:28])[0]
            m["lev"] = val if sc in (0, 255) else val / 10 ** _sm(bytes([sc]))
        elif num == 5:
            m["t5"] = struct.unpack(">H", s[9:11])[0]
            m["npts"] = struct.unpack(">I", s[5:9])[0]
            m["R"] = struct.unpack(">f", s[11:15])[0]
            m["E"], m["D"], m["nbits"] = _sm(s[15:17]), _sm(s[17:19]), s[19]
            m["rel5"] = p
        elif num == 6:
            m["bitmap"] = s[5]
        p += n


def read_header(url, off, size=4096):
    while True:
        buf = http_get(url, off, off + size - 1)
        m = parse_header(buf, off)
        if m is not None:
            return m
        if size >= 1 << 20:
            raise ValueError(f"GRIB header too long at byte {off} of {url}")
        size *= 8


def inventory(url):
    """the headers of all the messages of a GRIB2 file, reading only a few kB per message. The messages of one file
    usually have the same length: the next ones are guessed in groups of 64 and read in parallel; a wrong guess only
    means one more round."""
    from collections import Counter
    first = read_header(url, 0)
    length = _length(url)
    msgs, off, sizes, batch = [first], first["total"], Counter([first["total"]]), 8
    SL = 2048                                             # sizes may differ by a few bytes: look 2 kB around each guess
    while off < length:
        size = sizes.most_common(1)[0][0]                 # the usual length of a message in this file
        guesses = [off + k * size for k in range(batch) if off + k * size < length]
        bufs = pmap(lambda g: (max(0, g - SL), http_get(url, max(0, g - SL), g + 4095)), guesses)
        cur, got = off, 0
        for start, buf in bufs:
            rel = cur - start
            if rel < 0 or rel + 16 > len(buf) or buf[rel:rel + 4] != b"GRIB":
                break
            m = parse_header(buf[rel:], cur)
            if m is None:
                m = read_header(url, cur)
            msgs.append(m); cur += m["total"]; sizes[m["total"]] += 1; got += 1
            if cur >= length:
                break
        if got == 0:                                      # the guess failed at once: read the next header alone
            m = read_header(url, off)
            msgs.append(m); cur = off + m["total"]; sizes[m["total"]] += 1
        batch = min(64, 2 * batch) if got == len(guesses) else max(4, 2 * got)
        off = cur
    return msgs


_LEN = {}
def _length(url):
    """total size of a file, from the Content-Range of a 1-byte request"""
    if url not in _LEN:
        with HTTP.get(url, headers={"Range": "bytes=0-0"}, timeout=30, stream=True) as r:
            if r.status_code != 206 or "Content-Range" not in r.headers:
                raise RuntimeError(f"{urlparse(url).netloc} does not accept byte ranges (HTTP {r.status_code})")
            _LEN[url] = int(r.headers["Content-Range"].split("/")[1])
    return _LEN[url]


def mini_message(m):
    """the message without its data (a constant field): eccodes reads from it the grid, the time and the level"""
    head = bytearray(m["head"])
    head[m["rel5"] - 16 + 19] = 0                                   # 0 bits per value: no data section needed
    body = bytes(head) + b"\x00\x00\x00\x05\x07"
    return b"GRIB\x00\x00" + bytes([m["disc"], 2]) + struct.pack(">Q", 16 + len(body) + 4) + body + b"7777"


_GRIDS = {}
def grid_point(m, lat, lon):
    """nearest grid point to (lat, lon): index, its latitude and longitude, and the angle of the grid x axis
    from the east (for winds given relative to the grid)"""
    key = hashlib.md5(m["sec3"] + f"{lat:.4f},{lon:.4f}".encode()).hexdigest()      # one entry per grid and place
    if key not in _GRIDS:
        h = eccodes.codes_new_from_message(mini_message(m))
        try:
            la = eccodes.codes_get_array(h, "latitudes"); lo = eccodes.codes_get_array(h, "longitudes")
            ni = eccodes.codes_get(h, "Ni") if eccodes.codes_is_defined(h, "Ni") else None
            neg = eccodes.codes_get(h, "iScansNegatively") if eccodes.codes_is_defined(h, "iScansNegatively") else 0
            gtype = eccodes.codes_get(h, "gridType")
        finally:
            eccodes.codes_release(h)
        lo = ((lo - lon + 180.0) % 360.0) - 180.0 + lon
        d2 = (la - lat) ** 2 + ((lo - lon) * np.cos(np.radians(lat))) ** 2
        k = int(np.argmin(d2))
        dist = 111.2 * float(np.sqrt(d2[k]))
        beta = 0.0
        if ni:                                                # direction of increasing x at the point
            step = 1 if (k % ni) < ni - 1 else -1
            dy, dx = la[k + step] - la[k], (lo[k + step] - lo[k]) * np.cos(np.radians(la[k]))
            beta = float(np.arctan2(dy, dx)) + (np.pi if (step < 0) != bool(neg) else 0.0)
        if dist > 30:
            raise ValueError(f"the place is outside the grid of this model (nearest point {dist:.0f} km away)")
        _GRIDS[key] = dict(k=k, lat=float(la[k]), lon=float(((lo[k] + 180) % 360) - 180), dist=dist, beta=beta, gtype=gtype)
    return _GRIDS[key]


def describe(m):
    """time and level of a message, from eccodes: valid time, start and end of the period (hours), type of step"""
    h = eccodes.codes_new_from_message(mini_message(m))
    try:
        eccodes.codes_set(h, "stepUnits", 1)
        out = dict(stepType=eccodes.codes_get(h, "stepType"), start=int(eccodes.codes_get(h, "startStep")),
                   end=int(eccodes.codes_get(h, "endStep")),
                   uv_grid=int(eccodes.codes_get(h, "uvRelativeToGrid")) if eccodes.codes_is_defined(h, "uvRelativeToGrid") else 0)
    finally:
        eccodes.codes_release(h)
    return out


def point_value(url, m, k):
    """value at grid index k: only the bytes of that point are downloaded (simple packing), else the whole message"""
    if m.get("t5") == 0 and m.get("bitmap") == 255:
        nb = m["nbits"]
        if nb == 0:
            X = 0
        else:
            bit = k * nb
            b0, b1 = bit // 8, (bit + nb - 1) // 8
            raw = http_get(url, m["s7"] + 5 + b0, m["s7"] + 5 + b1)
            X = (int.from_bytes(raw, "big") >> (8 * (b1 - b0 + 1) - (bit % 8) - nb)) & ((1 << nb) - 1)
        return (m["R"] + X * 2.0 ** m["E"]) / 10.0 ** m["D"]
    data = http_get(url, m["off"], m["off"] + m["total"] - 1, timeout=300)    # other packings: the whole field
    h = eccodes.codes_new_from_message(data)
    try:
        v = float(eccodes.codes_get_values(h)[k])
        return np.nan if v == eccodes.codes_get(h, "missingValue") else v
    finally:
        eccodes.codes_release(h)


def field_max(url, m):
    """largest value of a whole field (used once, to check the units of the bundle precipitation)"""
    data = http_get(url, m["off"], m["off"] + m["total"] - 1, timeout=300)
    h = eccodes.codes_new_from_message(data)
    try:
        return float(np.nanmax(eccodes.codes_get_values(h)))
    finally:
        eccodes.codes_release(h)


# ================================================================ one model at the place
def latest_nwp_run(folder, names):
    """the latest run whose folders hold a GRIB file for every variable"""
    base = f"{METEOHUB}/nwp/{folder}/"
    runs = sorted({x.strip("/") for x in listing(base) if re.fullmatch(r"\d{10}/?", x)}, reverse=True)
    for run in runs[:4]:
        files = {}
        for v in names:
            g = [x for x in listing(f"{base}{run}/{v}/") if x.endswith(".grib")]
            if not g:
                break
            files[v] = f"{base}{run}/{v}/{g[0]}"
        else:
            return dt.datetime.strptime(run, "%Y%m%d%H"), files
    raise RuntimeError(f"no complete run of {folder} on MeteoHub (runs seen: {runs[:4]})")


def latest_bundle(dataset):
    page = http_get(f"{METEOHUB}/api/datasets/{dataset}/opendata", timeout=30)
    items = json.loads(page) if page else []
    if not items:
        raise RuntimeError(f"no open-data bundle of {dataset} on MeteoHub at the moment")
    it = max(items, key=lambda x: (x["date"], x["run"]))
    run = dt.datetime.strptime(f"{it['date']} {it['run']}", "%Y-%m-%d %H:%M")
    return run, f"{METEOHUB}/api/opendata/{it['filename']}"


def model_at_point(name, lat, lon):
    """{variable: {(start, end) or valid hour: value}} of the latest run of one MeteoHub model, at the grid point
    nearest to (lat, lon)"""
    spec = MODELS[name]
    if spec["kind"] == "nwp":
        run, files = latest_nwp_run(spec["folder"], list(spec["vars"]))
        jobs = [(spec["vars"][v], url) for v, url in files.items()]
    else:
        run, url = latest_bundle(spec["dataset"])
        jobs = [(None, url)]
    fn = os.path.join(CACHE, f"{name}_{run:%Y%m%d%H}_{lat:.4f}_{lon:.4f}.json")
    if os.path.exists(fn):
        with open(fn) as f:
            out = json.load(f)
        out["run"] = dt.datetime.fromisoformat(out["run"])
        out["seconds"] = 0                                          # nothing downloaded now
        return out
    t0, b0 = time.time(), STATS["bytes"]
    msgs = []
    for var, url in jobs:
        for m in inventory(url):
            v = var or GRIB_VARS.get((m["disc"], m["cat"], m["num"], m["lev_type"], int(m["lev"])))
            if v is None or (v == "zs" and any(x[0] == "zs" for x in msgs)):     # ground height: one value is enough
                continue
            msgs.append((v, url, m))
    if not msgs:
        raise RuntimeError(f"{name}: none of the wanted fields in the files")
    finest = {}                                   # a bundle can hold a variable on two grids: keep the finer one
    for v, _, m in msgs:
        finest[v] = max(finest.get(v, 0), m["npts"])
    msgs = [x for x in msgs if x[2]["npts"] == finest[x[0]]]
    grids = [grid_point(m, lat, lon) for _, _, m in msgs]   # the same grid is computed only once
    names_ = [x[0] for x in msgs]
    g = grids[names_.index("t2")] if "t2" in names_ else grids[0]          # the grid point reported: that of the temperature
    desc = [describe(m) for _, _, m in msgs]
    vals = pmap(lambda x: point_value(x[0][1], x[0][2], x[1]["k"]), list(zip(msgs, grids)), workers=24)
    series, uvgrid = {}, set()
    for (v, url, m), d, x in zip(msgs, desc, vals):
        if v in ("u10", "v10"):
            uvgrid.add(d["uv_grid"])
        if v in ("tp", "tp_rate"):
            a, b = (d["start"], d["end"]) if d["stepType"] != "instant" else (0, d["end"])
            series.setdefault(v, {})[f"{a}-{b}"] = x
        else:
            series.setdefault(v, {})[str(d["end"])] = x
    notes = []
    if any(m.get("t5") != 0 or m.get("bitmap") != 255 for _, _, m in msgs):
        notes.append("compressed fields (GRIB2 complex packing): whole fields downloaded, not only the point")
    if "tp_rate" in series:                       # 'precipitation rate' in the bundles: kg m-2 s-1, or already mm?
        rate = [(u, mm) for v, u, mm in msgs if v == "tp_rate"]
        mx = max(field_max(*rate[i]) for i in sorted({len(rate) // 4, len(rate) // 2, 3 * len(rate) // 4}))
        as_rate = mx < 0.05                       # 0.05 kg m-2 s-1 = 180 mm/h: larger values can only be amounts in mm
        series["tp"] = {}
        for ab, x in series.pop("tp_rate").items():
            a, b = map(int, ab.split("-"))
            series["tp"][ab] = x * 3600.0 * (b - a) if as_rate else x
        notes.append("precipitation coded as rate: converted to mm" if as_rate else "precipitation coded as rate, values in mm")
    gu = grids[names_.index("u10")] if "u10" in names_ else g
    if "u10" in series and "v10" in series and 1 in uvgrid and gu["beta"]:     # winds along the grid axes -> east, north
        cb, sb = np.cos(gu["beta"]), np.sin(gu["beta"])
        for h in list(series["u10"]):
            if h in series["v10"]:
                u, v = series["u10"][h], series["v10"][h]
                series["u10"][h], series["v10"][h] = u * cb - v * sb, u * sb + v * cb
        notes.append(f"winds rotated from the model grid to east-north ({np.degrees(gu['beta']):.1f}°)")
    if "zs" in series:
        series["zs"] = float(np.nanmean(list(series["zs"].values())))
    out = dict(run=run.isoformat(), grid=g, series=series, notes=notes, source=sorted({u for _, u, _ in msgs}),
               mb=(STATS["bytes"] - b0) / 1e6, seconds=time.time() - t0)
    with open(fn, "w") as f:
        json.dump(out, f)
    out["run"] = run
    return out


# ================================================================ GFS, the global model, as a reference
# (the functions of the GFS point meteogram, unchanged)
MG_CACHE = os.path.join(CACHE, "gfs")
os.makedirs(MG_CACHE, exist_ok=True)
MG_HTTP = requests.Session()
MG_HTTP.headers["User-Agent"] = "multimodel-meteogram (personal use; python-requests)"
MG_STATS = {"bytes": 0, "answers": {}}
MG_FAILS = {}
MG_NOMADS = "https://nomads.ncep.noaa.gov"
MG_CLOUDS = [("NOAA Open Data on AWS", "https://noaa-gfs-bdp-pds.s3.amazonaws.com"),
             ("NOAA Open Data on Google Cloud", "https://storage.googleapis.com/global-forecast-system")]


def mg_get(url, headers=None, timeout=60, tries=2):
    """GET with retries; None if the server cannot be reached (after 2 connection failures it is skipped)"""
    host = urlparse(url).netloc
    for k in range(tries):
        if MG_FAILS.get(host, 0) >= 2:
            return None
        try:
            r = MG_HTTP.get(url, headers=headers or {}, timeout=timeout)
            MG_STATS["answers"][r.status_code] = MG_STATS["answers"].get(r.status_code, 0) + 1
            if r.status_code in (429, 502, 503, 504):
                time.sleep(4 * (k + 1)); continue
            MG_STATS["bytes"] += len(r.content)
            return r
        except requests.ReadTimeout:
            MG_STATS["answers"]["ReadTimeout"] = MG_STATS["answers"].get("ReadTimeout", 0) + 1
        except requests.RequestException as err:
            name = type(err).__name__
            MG_STATS["answers"][name] = MG_STATS["answers"].get(name, 0) + 1
            MG_FAILS[host] = MG_FAILS.get(host, 0) + 1
            time.sleep(2)
    return None


def mg_file(key, ext):
    return os.path.join(MG_CACHE, hashlib.md5(key.encode()).hexdigest()[:20] + ext)

# ---------------------------------------------------------------- GFS at one point
MG_VARS = ("PRMSL", "TMP", "DPT", "RH", "UGRD", "VGRD", "GUST", "APCP", "HGT")
MG_LEVS = ("mean_sea_level", "2_m_above_ground", "10_m_above_ground", "surface")
MG_WANT = {("PRMSL", "mean sea level"), ("TMP", "2 m above ground"), ("DPT", "2 m above ground"),
           ("RH", "2 m above ground"), ("UGRD", "10 m above ground"), ("VGRD", "10 m above ground"),
           ("GUST", "surface"), ("APCP", "surface"), ("HGT", "surface")}
MG_NAMES = {("prmsl", "meanSea"): "mslp", ("2t", "heightAboveGround"): "t2", ("2d", "heightAboveGround"): "td2",
            ("2r", "heightAboveGround"): "rh2", ("10u", "heightAboveGround"): "u10", ("10v", "heightAboveGround"): "v10",
            ("gust", "surface"): "gust", ("orog", "surface"): "zs", ("tp", "surface"): "tp"}
def mg_latest_run(last_fh):
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    c0 = now.replace(minute=0, second=0, microsecond=0, hour=now.hour - now.hour % 6)
    for k in range(12):
        run = c0 - dt.timedelta(hours=6 * k)
        path = f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z."
        urls = [f"{MG_NOMADS}/pub/data/nccf/com/gfs/prod/{path}pgrb2.0p25.f{last_fh:03d}.idx"]
        urls += [f"{base}/{path}pgrb2.0p50.f{last_fh:03d}.idx" for _, base in MG_CLOUDS]
        for u in urls:
            r = mg_get(u, timeout=30)
            if r is not None and r.status_code == 200 and b":d=" in r.content[:200]:
                return run
    raise RuntimeError(f"No recent GFS run found. Answers of the servers (HTTP status: count): {MG_STATS['answers']}")


def mg_nomads(run, fh, lat, lon, d=0.5):
    """a small box (±0.5°) around the point from the NOMADS GRIB filter, 0.25° grid, surface fields only"""
    S_, N_ = max(lat - d, -90), min(lat + d, 90)
    for left, right in ((lon - d, lon + d), ((lon - d) % 360, (lon + d) % 360)):
        q = [("dir", f"/gfs.{run:%Y%m%d}/{run:%H}/atmos"), ("file", f"gfs.t{run:%H}z.pgrb2.0p25.f{fh:03d}")]
        q += [(f"var_{v}", "on") for v in MG_VARS] + [(f"lev_{l}", "on") for l in MG_LEVS]
        q += [("subregion", ""), ("leftlon", round(left, 3)), ("rightlon", round(right, 3)), ("toplat", round(N_, 3)),
              ("bottomlat", round(S_, 3))]
        url = f"{MG_NOMADS}/cgi-bin/filter_gfs_0p25.pl?" + urlencode(q)
        fn = mg_file(url, ".grib2")
        if os.path.exists(fn):
            return fn
        r = mg_get(url, timeout=60)
        time.sleep(1)                                         # NOMADS limits the number of requests per minute
        if r is not None and r.status_code == 200 and r.content[:4] == b"GRIB":
            with open(fn, "wb") as f:
                f.write(r.content)
            return fn
    return None


def mg_cloud(run, fh):
    """the same fields, whole globe at 0.5°, from NOAA Open Data (AWS, then Google Cloud), with HTTP byte ranges"""
    for label, base in MG_CLOUDS:
        url = f"{base}/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p50.f{fh:03d}"
        fn = mg_file(url + "meteogram", ".grib2")
        if os.path.exists(fn):
            return fn, label
        r = mg_get(url + ".idx", timeout=60)
        if r is None or r.status_code != 200:
            continue
        rows = [l.split(":") for l in r.text.strip().splitlines() if l.count(":") >= 6]
        parts = []
        for k, row in enumerate(rows):
            if (row[3], row[4]) in MG_WANT:
                end = str(int(rows[k + 1][1]) - 1) if k + 1 < len(rows) else ""
                rr = mg_get(url, headers={"Range": f"bytes={row[1]}-{end}"}, timeout=120)
                if rr is None or rr.status_code not in (200, 206):
                    parts = None; break
                parts.append(rr.content)
        if parts:
            with open(fn, "wb") as f:
                f.write(b"".join(parts))
            return fn, label
    return None, None

def bilinear(F, la, lo, y, x):
    """value at latitude y, longitude x from the four grid points around it (NaN outside the grid)"""
    lo = ((np.asarray(lo, float) - x + 180.0) % 360.0) - 180.0 + x      # longitudes within 180° of x, any convention
    io_, jo = np.argsort(lo), np.argsort(la)
    lo, la, F = lo[io_], np.asarray(la, float)[jo], np.asarray(F, float)[jo][:, io_]
    i = int(np.clip(np.searchsorted(lo, x, side="right") - 1, 0, lo.size - 2))
    j = int(np.clip(np.searchsorted(la, y, side="right") - 1, 0, la.size - 2))
    if not (lo[i] - 1e-6 <= x <= lo[i + 1] + 1e-6 and la[j] - 1e-6 <= y <= la[j + 1] + 1e-6):
        return np.nan
    wx, wy = (x - lo[i]) / (lo[i + 1] - lo[i]), (y - la[j]) / (la[j + 1] - la[j])
    return float((1 - wy) * ((1 - wx) * F[j, i] + wx * F[j, i + 1]) + wy * ((1 - wx) * F[j + 1, i] + wx * F[j + 1, i + 1]))


def mg_point(path, lat, lon):
    """values at the point of every wanted field in a GRIB2 file; precipitation as {(start, end): mm}"""
    vals, tp = {}, {}
    with open(path, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            try:
                key = MG_NAMES.get((eccodes.codes_get(h, "shortName"), eccodes.codes_get(h, "typeOfLevel")))
                if key is None:
                    continue
                ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
                la = eccodes.codes_get_array(h, "latitudes").reshape(nj, ni)[:, 0]
                lo = eccodes.codes_get_array(h, "longitudes").reshape(nj, ni)[0, :]
                F = eccodes.codes_get_values(h).reshape(nj, ni)
                F = np.where(F == eccodes.codes_get(h, "missingValue"), np.nan, F)
                v = bilinear(F, la, lo, lat, lon)
                if key == "tp":
                    tp[(int(eccodes.codes_get(h, "startStep")), int(eccodes.codes_get(h, "endStep")))] = v
                else:
                    vals[key] = v
            finally:
                eccodes.codes_release(h)
    return vals, tp


def gfs_at_point(lat, lon, hours):
    """the latest GFS run at the point, every 3 hours, in the same form as the MeteoHub models"""
    run = mg_latest_run(hours)
    series, srcs, use_nomads = {}, set(), True
    b0, t0 = MG_STATS["bytes"], time.time()
    for fh in range(0, hours + 1, 3):
        fn = mg_nomads(run, fh, lat, lon) if use_nomads else None
        src = "NOMADS GRIB filter (NOAA/NCEP), 0.25°"
        if not fn:
            use_nomads = False
            fn, label = mg_cloud(run, fh)
            src = f"{label}, 0.5°"
        if not fn:
            if fh == 0:
                raise RuntimeError(f"no GFS data for the run {run:%Y-%m-%d %H} UTC (answers of the servers: {MG_STATS['answers']})")
            break
        v, tp = mg_point(fn, lat, lon)
        for key, x in v.items():
            series.setdefault(key, {})[str(fh)] = x
        for (a, b), x in tp.items():
            series.setdefault("tp", {})[f"{a}-{b}"] = x
        srcs.add(src)
        print(f"+{fh}", end=" ", flush=True)
    print()
    if "zs" in series:
        series["zs"] = float(np.nanmean(list(series["zs"].values())))
    return dict(run=run, grid=dict(dist=0.0, gtype="regular_ll, bilinear interpolation"), series=series, notes=[],
                source=sorted(srcs), mb=(MG_STATS["bytes"] - b0) / 1e6, seconds=time.time() - t0)


# ================================================================ from raw values to curves
def curve(raw, key, run):
    """(valid times, values) of an instantaneous variable, sorted, without missing values"""
    d = raw["series"].get(key) or {}
    if not isinstance(d, dict) or not d:
        return np.array([], "datetime64[s]").astype(object), np.array([])
    hs = sorted(int(h) for h in d)
    v = np.array([d[str(h)] if d[str(h)] is not None else np.nan for h in hs], float)
    t = np.array([run + dt.timedelta(hours=h) for h in hs])
    ok = np.isfinite(v)
    return t[ok], v[ok]


def cumulative(rec):
    """accumulated precipitation since the start of the run at the end of each period, from records 'a-b': mm, whatever
    the periods are (0-1, 0-2 ... or 0-3, 0-6, 6-9 ... or 0-1, 1-2 ...)"""
    iv = sorted(((int(a), int(b), x) for ab, x in rec.items() for a, b in [ab.split("-")] if x is not None and np.isfinite(x)),
                key=lambda r: (r[1], r[0]))
    if not iv:
        return {}
    C = {min(a for a, _, _ in iv): 0.0}
    for a, b, x in iv:
        if b not in C and a in C:
            C[b] = max(C[a], C[a] + x)                    # tiny negative values from rounding: no precipitation
    return C


def build(name, raw, style):
    run = raw["run"]
    S = dict(name=name, run=run, color=style["color"], res=style["res"], who=style["who"], raw=raw,
             ls="--" if name == "GFS" else "-", lw=1.8 if name == "GFS" else 2.0)
    for key in ("t2", "td2", "mslp", "u10", "v10", "gust", "rh2"):
        S[key] = curve(raw, key, run)
    S["t2"] = (S["t2"][0], S["t2"][1] - 273.15)
    S["td2"] = (S["td2"][0], S["td2"][1] - 273.15)
    if not len(S["td2"][1]) and len(S["rh2"][1]):                  # dew point from T and RH (Bolton 1980), if missing
        tt = dict(zip(S["t2"][0], S["t2"][1]))
        keep = [k for k, t in enumerate(S["rh2"][0]) if t in tt]
        T = np.array([tt[S["rh2"][0][k]] for k in keep]); RH = np.clip(S["rh2"][1][keep], 1, 100)
        lne = np.log(RH / 100 * np.exp(17.67 * T / (T + 243.5)))
        S["td2"] = (S["rh2"][0][keep], 243.5 * lne / (17.67 - lne))
    S["mslp"] = (S["mslp"][0], S["mslp"][1] / 100.0)
    tu = dict(zip(S["u10"][0], S["u10"][1])); tv = dict(zip(S["v10"][0], S["v10"][1]))
    tw = np.array(sorted(set(tu) & set(tv)))
    U = np.array([tu[t] for t in tw]); V = np.array([tv[t] for t in tw])
    S["speed"] = (tw, np.hypot(U, V))
    S["dir"] = (tw, (270.0 - np.degrees(np.arctan2(V, U))) % 360.0)          # direction the wind comes FROM
    C = cumulative(raw["series"].get("tp") or {})
    hs = sorted(C)
    S["pcum"] = (np.array([run + dt.timedelta(hours=h) for h in hs]), np.array([C[h] for h in hs]))
    S["zs"] = raw["series"].get("zs") if isinstance(raw["series"].get("zs"), float) else np.nan
    return S


def at(times, values, t):
    """linear interpolation of a curve at time t (NaN outside it)"""
    if len(times) < 2 or t < times[0] or t > times[-1]:
        return np.nan
    x = np.array([(s - times[0]).total_seconds() for s in times])
    return float(np.interp((t - times[0]).total_seconds(), x, values))


# ================================================================ the meteogram
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
def compass(deg):
    return COMPASS[int(((deg + 22.5) % 360) // 45)]


def plot_multimodel(models, t0, t1):
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Europe/Rome")
    loc = lambda ts: [x.replace(tzinfo=dt.timezone.utc).astimezone(tz).replace(tzinfo=None) for x in ts]
    nm = len(models)                                            # one hourly-precipitation histogram per model
    ratios = [1.0, 1.25, 0.9, 1.0, 0.7, 0.2 + 0.17 * nm] + [0.6] * nm + [1.1]
    fig, axs = plt.subplots(len(ratios), 1, figsize=(13.5, 23 + 2.0 * nm), sharex=True,
                            gridspec_kw=dict(height_ratios=ratios, hspace=0.42))
    a_p, a_t, a_d, a_w, a_dir, a_h = axs[:6]
    a_bars, a_r = list(axs[6:6 + nm]), axs[-1]
    win = lambda T: np.array([t0 <= x <= t1 for x in T], bool) if len(T) else np.array([], bool)
    # range of the models for the temperature: on a common hourly axis, where at least 2 models have a value
    hours = [t0 + dt.timedelta(hours=k) for k in range(int((t1 - t0).total_seconds() // 3600) + 1)]
    grid = np.array([[at(*S["t2"], h) for h in hours] for S in models])
    n_ok = np.isfinite(grid).sum(0)
    if len(models) > 1:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # hours with no model: NaN, no band
            lo_ = np.where(n_ok >= 2, np.nanmin(grid, 0), np.nan); hi_ = np.where(n_ok >= 2, np.nanmax(grid, 0), np.nan)
        a_t.fill_between(loc(hours), lo_, hi_, color="#bbbbbb", alpha=0.35, lw=0, label="range of the models")
    for S in models:
        kw = dict(color=S["color"], ls=S["ls"], lw=S["lw"])
        for ax, key in ((a_p, "mslp"), (a_t, "t2"), (a_d, "td2"), (a_w, "speed")):
            T, V = S[key]
            w = win(T)
            if w.any():
                ax.plot(loc(T[w]), V[w], **kw)
        T, V = S["gust"]; w = win(T)
        if w.any():
            a_w.plot(loc(T[w]), V[w], color=S["color"], ls=":", lw=1.2)
        T, D = S["dir"]; w = win(T)
        if w.any():
            sp = S["speed"][1][w]
            a_dir.scatter(loc(T[w]), D[w], s=np.where(sp < 1.0, 10, 22), color=np.where(sp < 1.0, "#bbbbbb", S["color"]),
                          marker="s" if S["name"] == "GFS" else "o", zorder=3, lw=0)
        T, P = S["pcum"]
        if len(T) >= 2 and T[0] <= t0 <= T[-1]:
            base = at(T, P, t0)
            ts = [t0] + [x for x in T if t0 < x <= t1]
            a_r.plot(loc(ts), [at(T, P, x) - base for x in ts], **kw, drawstyle="default")
    a_p.set_ylabel("sea-level\npressure (hPa)")
    a_t.set_ylabel("temperature\nat 2 m (°C)")
    a_d.set_ylabel("dew point\nat 2 m (°C)")
    a_w.set_ylabel("wind at 10 m\n(m s⁻¹)")
    kmh = a_w.secondary_yaxis("right", functions=(lambda v: v * 3.6, lambda v: v / 3.6)); kmh.set_ylabel("km h⁻¹")
    a_w.set_ylim(bottom=0)
    a_w.set_title("solid: mean wind · dotted: gusts", loc="right", fontsize=10, pad=3, color="#555555")
    a_dir.set_ylim(-10, 370); a_dir.set_yticks([0, 90, 180, 270, 360]); a_dir.set_yticklabels(["N", "E", "S", "W", "N"])
    a_dir.set_ylabel("wind comes\nfrom")
    a_dir.set_title("grey dots: wind below 1 m s⁻¹ (direction not meaningful)", loc="right", fontsize=10, pad=3, color="#555555")
    # hourly precipitation: one row per model, one box per period of the model (1 h; 3 h for GFS), colour = mm per hour
    from matplotlib.colors import BoundaryNorm, ListedColormap
    levels = [0.1, 0.5, 1, 2, 5, 10, 20, 1000]
    cmap = ListedColormap(["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#7a0177", "#c51b8a"])
    norm = BoundaryNorm(levels, cmap.N)
    for i, S in enumerate(models):
        T, P = S["pcum"]
        for a, b, pa, pb in zip(T[:-1], T[1:], P[:-1], P[1:]):
            lo_t, hi_t = max(a, t0), min(b, t1)
            rate = (pb - pa) / ((b - a).total_seconds() / 3600)
            if hi_t <= lo_t or rate < 0.1:
                continue
            x0, x1 = mdates.date2num(loc([lo_t, hi_t]))
            a_h.add_patch(plt.Rectangle((x0, i - 0.42), x1 - x0, 0.84, color=cmap(norm(rate)), lw=0))
        a_h.axhline(i + 0.5, color="#dddddd", lw=0.6)
    a_h.set_ylim(len(models) - 0.5, -0.5)
    a_h.set_yticks(range(len(models))); a_h.set_yticklabels([S["name"] for S in models], fontsize=10)
    for lab, S in zip(a_h.get_yticklabels(), models):
        lab.set_color(S["color"])
    a_h.set_ylabel("precipitation\n(mm per hour)")
    a_h.set_title("each box: one hour (GFS: 3 hours, mean per hour); white: less than 0.1 mm per hour",
                  loc="right", fontsize=10, pad=3, color="#555555")
    sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm_, cax=a_h.inset_axes([1.01, 0.0, 0.012, 1.0]), ticks=levels[:-1])   # outside: same width as the others
    cb.ax.set_yticklabels(["0.1", "0.5", "1", "2", "5", "10", "20"], fontsize=8); cb.set_label("mm h⁻¹", fontsize=9)
    # hourly precipitation, one histogram per model, all with the same vertical scale
    top = 1.0
    for S in models:
        T, P = S["pcum"]
        for a, b, pa, pb in zip(T[:-1], T[1:], P[:-1], P[1:]):
            if b > t0 and a < t1:
                top = max(top, (pb - pa) / ((b - a).total_seconds() / 3600))
    for ax, S in zip(a_bars, models):
        T, P = S["pcum"]
        tot, has = 0.0, False
        for a, b, pa, pb in zip(T[:-1], T[1:], P[:-1], P[1:]):
            lo_t, hi_t = max(a, t0), min(b, t1)
            if hi_t <= lo_t:
                continue
            has = True
            h = (b - a).total_seconds() / 3600
            rate = max(0.0, (pb - pa) / h)
            tot += rate * (hi_t - lo_t).total_seconds() / 3600
            if rate > 0:
                x0, x1 = mdates.date2num(loc([lo_t, hi_t]))
                ax.bar(x0, rate, width=x1 - x0, align="edge", color=S["color"], alpha=0.85, lw=0.4, edgecolor="white")
        ax.set_ylim(0, top * 1.15)
        ax.set_ylabel(f"{S['name']}\n(mm h⁻¹)", color=S["color"])
        msg = (f"{tot:.1f} mm in the hours shown" if tot >= 0.05 else "no rain in the hours shown") if has else "no data in these hours"
        ax.text(0.995, 0.93, msg + (" · bars of 3 h, mean per hour" if S["name"] == "GFS" else " · bars of 1 h"),
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#444444",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    a_r.set_ylabel("precipitation\nsince the start (mm)")
    a_r.set_ylim(bottom=0, top=max(2.0, a_r.get_ylim()[1]))
    a_t.axhline(0, color="#777777", lw=0.8, ls=":"); a_d.axhline(0, color="#777777", lw=0.8, ls=":")
    for ax in axs:
        ax.grid(True, axis="x", which="major", color="#cccccc", lw=0.6)
        ax.grid(True, axis="x", which="minor", color="#e4e4e4", lw=0.5)
        if ax not in (a_dir, a_h):
            ax.grid(True, axis="y", color="#eeeeee", lw=0.6)
    L0, L1 = loc([t0, t1])
    short = (t1 - t0) <= dt.timedelta(hours=24)               # short windows: ticks every 3 h and every hour
    major = range(0, 24, 3) if short else [0, 6, 12, 18]
    a_r.xaxis.set_major_locator(mdates.HourLocator(byhour=major))
    a_r.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1 if short else 3)))
    E0 = L0 - dt.timedelta(hours=1)                            # left edge of the axis
    E0 = E0.replace(minute=0, second=0, microsecond=0) + (dt.timedelta(hours=1) if E0.minute or E0.second else dt.timedelta(0))
    first = next(E0 + dt.timedelta(hours=k) for k in range(25) if (E0 + dt.timedelta(hours=k)).hour in major)
    def _fmt(x, _):
        d = mdates.num2date(x).replace(tzinfo=None)
        same_first = abs((d - first).total_seconds()) < 60      # the date also under the first tick
        return d.strftime("%H\n%a %d %b") if d.hour == 0 or same_first else d.strftime("%H")
    a_r.xaxis.set_major_formatter(plt.FuncFormatter(_fmt))
    a_r.set_xlim(mdates.date2num(L0) - 1 / 24, mdates.date2num(L1) + 1 / 24)
    a_r.set_xlabel("Italian time (Europe/Rome): hours, and the date at 00 and at the start")
    for ax in axs:                                              # hours and dates under every panel
        ax.tick_params(axis="x", which="both", labelbottom=True, labelsize=10)
    for x in [d for d in (L0 + dt.timedelta(hours=k) for k in range(0, int((L1 - L0).total_seconds() // 3600) + 1)) if d.hour == 0]:
        for ax in axs:
            ax.axvline(x, color="#888888", lw=0.9, zorder=0)
    now = loc([dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)])[0]
    if L0 <= now <= L1:
        for ax in axs:
            ax.axvline(now, color="#e377c2", lw=1.2, ls="--", zorder=1)
    handles = [plt.Line2D([], [], color=S["color"], ls=S["ls"], lw=2.5) for S in models]
    labels = [f"{S['name']} · {S['res']} · run {S['run']:%d/%m %H} UTC" for S in models]
    if len(models) > 1:
        handles.append(plt.Rectangle((0, 0), 1, 1, color="#bbbbbb", alpha=0.5)); labels.append("range of the models (temperature)")
    if L0 <= now <= L1:
        handles.append(plt.Line2D([], [], color="#e377c2", ls="--", lw=1.2)); labels.append("now")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.085, 0.978), ncol=3, fontsize=10.5, frameon=False)
    fig.suptitle(f"{PLACE['name']} · multi-model meteogram · {PLACE['lat']:.3f}° N, {PLACE['lon']:.3f}° E · real ground {PLACE['z']} m",
                 x=0.01, ha="left", y=0.995, fontsize=15)
    fig.text(0.01, 0.005, "data: MeteoHub open data, ItaliaMeteo (ICON-2I, WRF, MOLOCH, BOLAM; terms at meteohub.agenziaitaliameteo.it/app/license); NOAA/NCEP (GFS). "
             "MeteoHub models: value of the grid point nearest to the place; GFS: bilinear from the four points around it.\n"
             "Every curve is one forecast of one run; the runs start at different times. Precipitation: amount accumulated "
             "from the first time shown, so that all the curves start from zero together.", fontsize=9, color="#555555", va="bottom")
    fig.subplots_adjust(left=0.1, right=0.93, top=0.93, bottom=0.055)
    fig.align_ylabels(axs)
    return fig


def summary(models, t0, t1):
    """one line per model, computed from the same numbers as the meteogram, in a common window"""
    ends = [S["pcum"][0][-1] for S in models if len(S["pcum"][0]) >= 2 and S["pcum"][0][0] <= t0 < S["pcum"][0][-1]]
    tp_end = min([t1] + ends)
    print(f"\nsummary from {t0:%a %d %b %H} UTC to {t1:%a %d %b %H} UTC (precipitation up to {tp_end:%a %d %b %H} UTC, "
          "the last time common to all the models):")
    print(f"{'model':9} {'run (UTC)':12} {'grid':>9} {'point':>8} {'ground':>7} {'T min':>6} {'T max':>6} {'precip':>7} {'max gust':>9}")
    rows = []
    for S in models:
        T, V = S["t2"]; w = np.array([t0 <= x <= t1 for x in T], bool)
        tmin, tmax = (np.min(V[w]), np.max(V[w])) if w.any() else (np.nan, np.nan)
        P = at(*S["pcum"], tp_end) - at(*S["pcum"], t0) if len(S["pcum"][0]) >= 2 and tp_end > t0 else np.nan
        G, Gv = S["gust"]; wg = np.array([t0 <= x <= t1 for x in G], bool)
        gmax = np.max(Gv[wg]) * 3.6 if wg.any() else np.nan
        dist = S["raw"]["grid"].get("dist", np.nan)
        f = lambda x, fmt: (fmt % x) if np.isfinite(x) else "n.a."
        print(f"{S['name']:9} {S['run']:%d/%m %H}     {S['res'].split(',')[0]:>9} {f(dist, '%.1f km'):>8} {f(S['zs'], '%.0f m'):>7} "
              f"{f(tmin, '%.1f'):>6} {f(tmax, '%.1f'):>6} {f(P, '%.1f mm'):>7} {f(gmax, '%.0f km/h'):>9}")
        rows.append(dict(model=S["name"], tmin=tmin, tmax=tmax, precip=P, gust=gmax))
    t_all = [r for r in rows if np.isfinite(r["tmax"])]
    if len(t_all) > 1:
        print(f"spread of the models: highest temperature {min(r['tmax'] for r in t_all):.1f} to {max(r['tmax'] for r in t_all):.1f} °C; "
              f"lowest {min(r['tmin'] for r in t_all):.1f} to {max(r['tmin'] for r in t_all):.1f} °C", end="")
        p_all = [r["precip"] for r in rows if np.isfinite(r["precip"])]
        print(f"; precipitation {min(p_all):.1f} to {max(p_all):.1f} mm" if len(p_all) > 1 else "")
    print(f"'point': distance of the grid point used from the place; 'ground': height of the model ground there "
          f"({PLACE['name']}: about {PLACE['z']} m; a model ground 100 m lower is about 0.65 °C warmer)")


def save_csv(models, fn):
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "run_utc", "time_utc", "variable", "value", "unit"])
        units = dict(mslp="hPa", t2="degC", td2="degC", speed="m/s", gust="m/s", dir="deg_from", pcum="mm_since_run_start")
        for S in models:
            for key, u in units.items():
                for t, v in zip(*S[key]):
                    w.writerow([S["name"], f"{S['run']:%Y-%m-%d %H:%M}", f"{t:%Y-%m-%d %H:%M}", key, f"{v:.2f}", u])


def make_meteogram(names=("ICON-2I", "WRF", "MOLOCH", "BOLAM", "GFS"), hours=72, place=None, start="run", outdir=".", show=True):
    """start="run": the window starts at the most recent run among the models; start="now": at the current hour"""
    """downloads the latest run of each chosen model at the place, draws the meteogram, saves PNG and CSV, prints a summary"""
    if place:
        PLACE.clear(); PLACE.update(name=place, **PLACES[place])
    lat, lon = PLACE["lat"], PLACE["lon"]
    print(f"place: {PLACE['name']}, {lat:.4f}° N, {lon:.4f}° E, real ground {PLACE['z']} m")
    models = []
    for name in names:
        print(f"{name}: latest run ...", flush=True)
        try:
            if name == "GFS":
                raw = gfs_at_point(lat, lon, min(96, int(hours) + (12 if start == "now" else 0)))
                style = GFS_STYLE
            else:
                raw = model_at_point(name, lat, lon)
                style = MODELS[name]
            S = build(name, raw, style)
            if not len(S["t2"][0]):
                raise RuntimeError("no 2 m temperature at the place in the files")
            last = max(x[0][-1] for x in (S["t2"], S["pcum"]) if len(x[0]))
            print(f"  {name}: run {S['run']:%Y-%m-%d %H} UTC, up to +{int((last - S['run']).total_seconds() // 3600)} h"
                  + (f", grid point {raw['grid']['dist']:.1f} km from the place" if name != "GFS" else "")
                  + (f", {raw['mb']:.1f} MB in {raw['seconds']:.0f} s" if raw.get("seconds") else ", already downloaded in this session")
                  + ("; " + "; ".join(raw["notes"]) if raw["notes"] else ""))
            models.append(S)
        except Exception as err:
            print(f"  {name}: not available now ({err}); the meteogram is drawn without it")
    if not models:
        raise RuntimeError("no model could be downloaded")
    if start == "now":                                          # from the current hour ...
        t0 = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    else:                                                       # ... or from the start of the most recent run
        t0 = max(S["run"] for S in models)
    t1 = t0 + dt.timedelta(hours=int(hours))                    # for the chosen number of hours
    keep = []
    for S in models:                                            # a run that ends before the window: nothing to draw
        if len(S["t2"][0]) and S["t2"][0][-1] > t0:
            keep.append(S)
        else:
            print(f"  {S['name']}: not used, its forecast ends before {t0:%d/%m %H} UTC")
    if not keep:
        raise RuntimeError("no model covers the hours chosen")
    models = keep
    fig = plot_multimodel(models, t0, t1)
    tag = f"{PLACE['tag']}_{t0:%Y%m%d%H}"
    png, csvf = os.path.join(outdir, f"meteogram_{tag}.png"), os.path.join(outdir, f"meteogram_{tag}.csv")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    save_csv(models, csvf)
    summary(models, t0, t1)
    print(f"files saved: {png} and {csvf}")
    return models, png, t0, t1


# ================================================================ command line
def main():
    import argparse, contextlib, glob, io, shutil
    ap = argparse.ArgumentParser(description="Rome multi-model meteogram")
    ap.add_argument("--hours", type=int, default=15, help="hours shown, from the current hour (default 15)")
    ap.add_argument("--models", default="ICON-2I,WRF,MOLOCH,BOLAM,GFS", help="comma-separated list")
    ap.add_argument("--out", default="meteograms", help="folder for the PNG and CSV files")
    ap.add_argument("--keep", type=int, default=14, help="days of meteograms kept in the folder (default 14)")
    ap.add_argument("--readme", default="README.md", help="page updated with the latest meteogram ('' for none)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = io.StringIO()

    class Tee(io.TextIOBase):                             # print to the job log and keep a copy for the README
        def write(self, x):
            sys.__stdout__.write(x); log.write(x); return len(x)
    with contextlib.redirect_stdout(Tee()):
        models, png, t0, t1 = make_meteogram([m.strip() for m in a.models.split(",") if m.strip()], a.hours,
                                             start="now", outdir=a.out, show=False)
    shutil.copyfile(png, os.path.join(a.out, "latest.png"))
    shutil.copyfile(png[:-4] + ".csv", os.path.join(a.out, "latest.csv"))
    # keep only the last days
    old = sorted(glob.glob(os.path.join(a.out, "meteogram_*.png")))[:-a.keep or None] if a.keep else []
    for f in old:
        for g in (f, f[:-4] + ".csv"):
            if os.path.exists(g):
                os.remove(g)
    if a.readme:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Europe/Rome")
        L = lambda t: t.replace(tzinfo=dt.timezone.utc).astimezone(tz)
        text = log.getvalue()
        summ = text[text.find("summary from"):].split("files saved")[0].strip() if "summary from" in text else ""
        notes = [l.strip() for l in text.splitlines() if "not available now" in l or "not used," in l]
        kept = sorted(glob.glob(os.path.join(a.out, "meteogram_*.png")), reverse=True)
        with open(a.readme, "w") as f:
            f.write(f"# Rome · multi-model meteogram\n\n")
            f.write(f"**{L(t0):%A %d %B %Y, %H:%M} → {L(t1):%H:%M}** (Italian time) · updated "
                    f"{dt.datetime.now(tz):%d/%m %H:%M}\n\n")
            f.write(f"![meteogram]({png})\n\n")
            if notes:
                f.write("**Missing today:** " + "; ".join(notes) + "\n\n")
            f.write("```\n" + summ + "\n```\n\n")
            f.write("Models: ICON-2I (2.2 km), WRF (1.5 km), MOLOCH (1.2 km), BOLAM (8 km) from MeteoHub "
                    "(ItaliaMeteo); GFS (25 km) from NOAA. Every curve is one forecast of one run.\n\n")
            days = []
            for k in kept[1:]:
                stamp = re.findall(r"(\d{10})", os.path.basename(k))[-1]
                days.append(f"[{stamp[6:8]}/{stamp[4:6]}]({k})")
            f.write("Previous days: " + (" · ".join(days) or "none yet") + "\n")
    print(f"done: {png}")


if __name__ == "__main__":
    main()
