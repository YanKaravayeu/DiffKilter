import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from sklearn.neighbors import KDTree

try:
    import pytesseract
except ImportError:
    pytesseract = None

#  Config paths 
CFG_DIR = Path(__file__).parent.resolve()
LAYOUT_PATH = CFG_DIR / "layout.json"      
CONFIG_PATH = CFG_DIR / "config.json"       #  Homography H, board_roi, swipe_y
COLORS_PATH = CFG_DIR / "colors.json"       #  HSV bands per role
RAW_LOG = CFG_DIR / "scrape.jsonl"
CHUNKS_DIR = CFG_DIR / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

def load_layout():
    if not LAYOUT_PATH.exists():
        raise FileNotFoundError("Missing layout.json")
    L = json.load(open(LAYOUT_PATH, "r", encoding="utf-8"))
    XY = np.array(L["xy"], dtype=np.float32)
    if XY.ndim != 2 or XY.shape[1] != 2:
        raise ValueError("layout.json 'xy' must be [N,2] normalized coords")
    return L, XY

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config.json at {CONFIG_PATH}. Run calibrate-h.")
    C = json.load(open(CONFIG_PATH, "r"))
    H = np.array(C["H"], dtype=np.float32)
    board_roi = tuple(C["board_roi"])
    swipe_y = int(C["swipe_y"])
    return H, board_roi, swipe_y

def load_colors():
    if not COLORS_PATH.exists():
        raise FileNotFoundError("Missing colors.json. Run calibrate-colors.")
    B = json.load(open(COLORS_PATH, "r"))
    # each role: {"lo":[H,S,V], "hi":[H,S,V]}
    return B

# ---------- ADB ----------
def adb_bin():
    env = os.environ.get("ADB_BIN")
    if env: return env
    #  Prefer HD-Adb.exe from BlueStacks if present; else 'adb' in PATH
    candidates = [
        "adb", # Prefer Linux adb
        r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
        r"C:\Program Files (x86)\BlueStacks_nxt\HD-Adb.exe",
        "/mnt/c/Program Files/BlueStacks_nxt/HD-Adb.exe",
        "/mnt/c/Program Files (x86)/BlueStacks_nxt/HD-Adb.exe",
        "adb",
    ]
    for c in candidates:
        if c == "adb" or os.path.exists(c):
            return c
    return "adb"

def adb(cmd: list[str]) -> bytes:
    exe = adb_bin()
    return subprocess.check_output([exe] + cmd, stderr=subprocess.STDOUT)

def adb_connect(addr="127.0.0.1:5555"):
    try:
        adb(["kill-server"])
    except Exception:
        pass
    adb(["start-server"])
    try:
        adb(["connect", addr])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ADB connect failed: {e.output.decode(errors='ignore')}")

def screencap_bgr() -> np.ndarray:
    try:
        png = adb(["exec-out", "screencap", "-p"])
    except subprocess.CalledProcessError:
        # fallback for older/quirky ADBs
        png = adb(["shell", "screencap", "-p"])
        png = png.replace(b"\r\n", b"\n")
    arr = np.frombuffer(png, np.uint8)
    bgr = cv.imdecode(arr, cv.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to decode screencap")
    return bgr


def swipe_next(x1, y, x2, dur_ms=120):
    """
    Simulates a physical swipe gesture on the Android emulator using ADB.
    
    Uses the Android Debug Bridge (ADB) shell command to input a swipe, 
    navigating to the next climbing route in the app's list.

    Args:
        x1 (int): The starting X pixel coordinate of the swipe.
        y (int): The Y pixel coordinate of the swipe.
        x2 (int): The ending X pixel coordinate of the swipe.
        dur_ms (int, optional): Duration of the swipe in milliseconds. Defaults to 120.
    """

    adb(["shell", "input", "swipe", str(x2), str(y), str(x1), str(y), str(dur_ms)])

def dump_uix() -> str | None:
    try:
        adb(["shell", "uiautomator", "dump", "/sdcard/view.xml"])
        xml = adb(["shell", "cat", "/sdcard/view.xml"]).decode("utf-8", "ignore")
        return xml
    except subprocess.CalledProcessError:
        return None

#  Geometry
def to_board_norm(px, py, H: np.ndarray):
    p = np.array([px, py, 1.0], dtype=np.float32)
    q = H @ p
    u, v = float(q[0]/q[2]), float(q[1]/q[2])  # (0..1, 0..1), origin = top-left
    return u, v

#  OCR / title+grade
def ocr_text(bgr, roi):
    if pytesseract is None:
        return ""
    x0,y0,x1,y1 = roi
    crop = bgr[y0:y1, x0:x1]
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 5, 40, 40)
    _, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return pytesseract.image_to_string(th, config="--psm 6")

def parse_title_grade(uix_xml, ocr_txt):
    title, grade = None, None
    texts = []
    if uix_xml:
        texts += re.findall(r'text="([^"]+)"', uix_xml)
    if ocr_txt:
        texts += [ln.strip() for ln in ocr_txt.splitlines()]
    texts = [t for t in texts if t and not t.isspace()]
    if texts:
        # Heuristic: choose the longest “natural language” line as title
        title = max(texts, key=len)[:128].strip()
    for t in texts:
        if re.search(r'\bV\d{1,2}\b', t) or re.search(r'\b[4-9][abc][+-]?\b', t, re.I):
            grade = t.strip()
            break
    return title, grade

#  Color/ring detection
def color_mask(hsv, lo, hi):
    return cv.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

def centers_from_mask(mask):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)
    cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    pts=[]
    for c in cnts:
        area = cv.contourArea(c)
        if 60 < area < 3500:
            (x,y),r = cv.minEnclosingCircle(c)
            pts.append((float(x), float(y), float(r)))
    return pts

def detect_rings(bgr, board_roi_px, bands):
    x0,y0,x1,y1 = board_roi_px
    crop = bgr[y0:y1, x0:x1]
    hsv  = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
    out = {"start":[], "hand":[], "finish":[], "feet":[]}
    for role in out.keys():
        if role not in bands: continue
        lo, hi = bands[role]["lo"], bands[role]["hi"]
        m = color_mask(hsv, lo, hi)
        for x,y,r in centers_from_mask(m):
            out[role].append({"px": x0+x, "py": y0+y, "r": r})
    return out

#  Snapping to LEDs
ROLE_ID = {"hand":1, "start":2, "finish":3, "feet":4}

def build_kdtree(XY):
    tree = KDTree(XY)
    d2, _ = tree.query(XY, k=2)
    r_gate = float(np.median(d2[:,1])) * 0.5  # gate = half of median NN distance
    return tree, r_gate

def normalize_detections(dets_px, H):
    out = {k: [] for k in dets_px}
    for role, arr in dets_px.items():
        for d in arr:
            u,v = to_board_norm(d["px"], d["py"], H)
            out[role].append({"u":u, "v":v})
    return out

def snap_to_leds(dets_uv, XY, tree, r_gate):
    tokens = np.zeros((len(XY),), dtype=np.uint8)
    for role in ("start","finish","feet","hand"):
        for p in dets_uv.get(role, []):
            dist, idx = tree.query([[p["u"], p["v"]]], k=1)
            if dist[0,0] <= r_gate:
                j = int(idx[0,0])
                tokens[j] = max(tokens[j], ROLE_ID[role])
    return tokens

#  End-of-list detectors
def route_fingerprint(name, grade, tokens):
    pts = list(np.nonzero(tokens)[0])
    s = f"{grade}|{name}|{pts}"
    return hashlib.sha1(s.encode()).hexdigest()

def phash_bits(bgr, roi):
    x0,y0,x1,y1 = roi
    crop = bgr[y0:y1, x0:x1]
    g = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    g = cv.resize(g, (32,32), interpolation=cv.INTER_AREA)
    g = np.float32(g)
    d = cv.dct(g)
    low = d[:8,:8]
    med = np.median(low[1:].ravel())
    bits = (low > med).astype(np.uint8).ravel()
    return bits

def hamming(a,b):
    return int(np.sum(a != b))

#  Persist
def save_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

class ChunkWriter:
    def __init__(self, layout_id, XY, chunk=500):
        self.layout_id = layout_id
        self.XY = XY
        self.chunk = chunk
        self.bufX, self.bufG, self.bufT = [], [], []
        self.i = 0
    def add(self, X, grade, title):
        self.bufX.append(X.astype(np.uint8))
        self.bufG.append(grade or "")
        self.bufT.append(title or "")
        if len(self.bufX) >= self.chunk:
            self.flush()
    def flush(self):
        if not self.bufX: return
        path = CHUNKS_DIR / f"{self.layout_id}_chunk_{self.i:04d}.npz"
        np.savez_compressed(path,
            X=np.stack(self.bufX,0),
            grades=np.array(self.bufG,object),
            titles=np.array(self.bufT,object),
            xy=self.XY
        )
        print(f"wrote {path} ({len(self.bufX)} routes)")
        self.bufX.clear(); self.bufG.clear(); self.bufT.clear()
        self.i += 1

#  Modes
def cmd_calibrate_h(args):
    """
    Calculates the Homography matrix to map screen pixels to board coordinates.

    Tool prompts user to click the four corners ofd the board in a screenshot.
    It then uses OpenCV to calculate Homography matrix and map raw pixels to a
    normalised (x, y) spatial coordinate.

    Args:
        args (argparse.Namespace): Command line arguments containing the path to
                                   the calibration screenshot.
    """

    img = cv.imread(args.screenshot)
    if img is None: raise RuntimeError("Could not read screenshot")
    clicks = []
    def cb(e,x,y,flags,param):
        if e == cv.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x,y))
    cv.namedWindow("calib"); cv.setMouseCallback("calib", cb)
    print("Click TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT. Press Enter to save.")
    while True:
        view = img.copy()
        for p in clicks: cv.circle(view, p, 6, (0,255,0), -1)
        cv.imshow("calib", view)
        k = cv.waitKey(20)
        if k == 13 and len(clicks)==4:  # Enter
            src = np.float32(clicks)
            dst = np.float32([[0,0],[1,0],[1,1],[0,1]])
            H = cv.getPerspectiveTransform(src, dst)
            x0 = min(p[0] for p in clicks); y0 = min(p[1] for p in clicks)
            x1 = max(p[0] for p in clicks); y1 = max(p[1] for p in clicks)
            swipe_y = int((y0+y1)//2)
            cfg = {"H": H.tolist(), "board_roi":[int(x0),int(y0),int(x1),int(y1)], "swipe_y": swipe_y}
            json.dump(cfg, open(CONFIG_PATH, "w"))
            print("Saved", CONFIG_PATH)
            break
        if k == 27: break  # Esc

def cmd_calibrate_colors(args):
    """
    Establishes HSV color boundaries for the Start, Hand, Foot, and Finish holds.
    
    Rather than hardcoding RGB pixel values, function allows the user to sample
    pixels directly. It converts them to HSV colour space and ensures robust hold
    detection via colour masking during the automated scraping loop.

    Args:
        args(argparse.Namespace): Command line arguments containing the path to the
                                  colour calibration screenshot.
    """
    
    img = cv.imread(args.screenshot)
    if img is None: raise RuntimeError("Could not read screenshot")
    roles = ["start","hand","finish","feet"]
    samples = {r: [] for r in roles}
    cur = 0
    def cb(e,x,y,flags,param):
        nonlocal cur
        if e == cv.EVENT_LBUTTONDOWN:
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            h,s,v = map(int, hsv[y,x])
            samples[roles[cur]].append((h,s,v))
            print(roles[cur], (h,s,v))
    cv.namedWindow("colors"); cv.setMouseCallback("colors", cb)
    print("Left-click ring pixels. Press Tab to switch role; Enter to save; Esc to abort.")
    while True:
        view = img.copy()
        cv.putText(view, f"Role: {roles[cur]}", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv.imshow("colors", view)
        k = cv.waitKey(20)
        if k == 9: cur = (cur+1)%len(roles)  # Tab
        if k == 13:  # Enter
            bands = {}
            for r, pts in samples.items():
                if not pts: continue
                arr = np.array(pts, dtype=np.int32)
                mu = arr.mean(0); sd = arr.std(0)
                lo = np.maximum(0, (mu - 2.5*sd)).astype(int).tolist()
                hi = np.minimum([179,255,255], (mu + 2.5*sd)).astype(int).tolist()
                bands[r] = {"lo": lo, "hi": hi}
            json.dump(bands, open(COLORS_PATH, "w"))
            print("Saved", COLORS_PATH)
            break
        if k == 27: break

def cmd_scrape(args):
    """
    Executes the automated scraping loop using ABD and Computer Vision.

    Connects to a running BlueStacks emulator, swipes through the Kilterboard
    app, captures screenshots, and uses HSV masking and OCR to extract hold 
    positions, route names, and grades.

    Args:
        args (argparse.Namespace): Command line arguments including ADB address.
    """
    
    (L, XY) = load_layout()
    H, board_roi, swipe_y = load_config()
    bands = load_colors()
    tree, r_gate = build_kdtree(XY)
    layout_id = L["layout_id"]

    #  ADB connect
    adb_connect(args.adb_addr)
    #  swipe coordinates
    x0,y0,x1,y1 = board_roi
    sx1 = x0 + int(0.25*(x1-x0))
    sx2 = x0 + int(0.75*(x1-x0))

    #  Chunk writer
    cw = ChunkWriter(layout_id, XY, chunk=args.chunk)

    #  End-of-list state
    seen_fp = set()
    prev_ph = None
    nochange = 0
    dup_hits = 0

    i = 0
    while i < args.max:
        bgr = screencap_bgr()
        uix = dump_uix()
        ocr = ocr_text(bgr, (x0, max(0,y0-160), x1, y0)) if args.ocr else ""
        title, grade = parse_title_grade(uix, ocr)

        det_px = detect_rings(bgr, board_roi, bands)
        det_uv = normalize_detections(det_px, H)
        tokens = snap_to_leds(det_uv, XY, tree, r_gate)

        #  Save
        rec = {
            "ts": time.time(), "title": title, "grade": grade,
            "tokens": tokens.tolist()
        }
        save_jsonl(RAW_LOG, rec)
        cw.add(tokens, grade, title)
        print(f"[{i}] {title or '?'} / {grade or '?'}  "
              f"nonzero={int(np.count_nonzero(tokens))}")

        #  End-of-list checks
        cur_ph = phash_bits(bgr, board_roi)
        if prev_ph is not None and hamming(prev_ph, cur_ph) <= 5:
            nochange += 1
        else:
            nochange = 0
        prev_ph = cur_ph

        fp = route_fingerprint(title or "", grade or "", tokens)
        if fp in seen_fp:
            dup_hits += 1
        else:
            dup_hits = 0
            seen_fp.add(fp)

        if nochange >= 3 or dup_hits >= 2:
            print("Likely end of list — stopping.")
            break

        #  Next
        swipe_next(sx1, swipe_y, sx2, dur_ms=120)
        time.sleep(args.sleep + random.random()*0.4)
        i += 1

    cw.flush()

#  Main
def main():
    p = argparse.ArgumentParser(description="Kilter route scraper → per-hold vectors")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("calibrate-h", help="click four board corners → save homography")
    a.add_argument("--screenshot", required=True, help="PNG from BlueStacks")
    a.set_defaults(func=cmd_calibrate_h)

    a = sub.add_parser("calibrate-colors", help="click ring pixels per role → save HSV bands")
    a.add_argument("--screenshot", required=True)
    a.set_defaults(func=cmd_calibrate_colors)

    a = sub.add_parser("scrape", help="run the scraper loop")
    a.add_argument("--adb-addr", default="127.0.0.1:5555")
    a.add_argument("--max", type=int, default=50000)
    a.add_argument("--sleep", type=float, default=0.9)
    a.add_argument("--chunk", type=int, default=500)
    a.add_argument("--ocr", action="store_true", help="use OCR fallback if needed")
    a.set_defaults(func=cmd_scrape)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()