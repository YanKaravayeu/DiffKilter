import json, argparse
from pathlib import Path
import numpy as np, cv2 as cv

BASE = Path(__file__).parent.resolve()
CONFIG_PATH = BASE / "config.json"
LAYOUT_PATH = BASE / "layout.json"

def to_board_norm(px, py, H):
    """
    Applies Homography to convert the screen pixels into normalised space.

    Takes a raw (x, y) pixel coordinate from the image, converts it to homogenous
    coordinates, multiplies it by the 3x3 Homography matrix 'H', and normalises the
    result back down to 2D space.

    Args:
        px (float): Raw X pixel coordinate on the image.
        py (float): Raw Y pixel coordinate on the image.
        H (np.ndarray): The 3x3 Homography matrix calculated via OpenCV.

    Returns:
        tuple[float, float]: The normalised (u, v) spatial coordinates (0.0 to 1.0).
    """

    p = np.array([px, py, 1.0], np.float32)
    q = H @ p
    return float(q[0]/q[2]), float(q[1]/q[2])

def parse_cols_seq(s, rows):
    """
    Parse shorthand string layout definition into a strict list of row lengths.

    Kilterboard has alternating hold pattern on the wall. Function accepts a compressed
    string formate like "17, 18x3, 17" and expands it into [17, 18, 18, 18, 17]. It also
    validates that the final count matches the expected number of rows.

    Args:
        s (str): The compressed sequence string (e.g. "17x2, 18x22).
        rows (int): The expected total number of rows for validation.

    Returns:
        list[int]: A fully expanded list of hold counts per row.

    Raises:
        SystemExit: If the expanded list length does not match the 'rows' parameter.
    """
    
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if "x" in tok.lower():
            v, r = tok.lower().split("x")
            out += [int(v)] * int(r)
        else:
            out.append(int(tok))
    if len(out) != rows:
        raise SystemExit(f"--cols-seq length {len(out)} != --rows {rows}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot", required=True)
    ap.add_argument("--layout-id", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols-seq", required=True,
                    help="per-row counts, e.g. '18,17x3,9,17...'; must match --rows")
    args = ap.parse_args()

    # load config (homography)
    C = json.load(open(CONFIG_PATH, "r"))
    H = np.array(C["H"], np.float32)

    img = cv.imread(args.screenshot)
    if img is None:
        raise SystemExit("Could not read --screenshot")

    cols_per_row = parse_cols_seq(args.cols_seq, args.rows)

    print("Click LEFT then RIGHT LED for each row, top → bottom.")
    print("Hotkeys:  z = undo last click,  Esc = quit,  Enter = finish (after 2*rows clicks).")

    clicks = []  # [(x,y), ...] length must be 2*rows
    def cb(e,x,y,flags,param):
        nonlocal clicks
        if e == cv.EVENT_LBUTTONDOWN:
            clicks.append((x,y))

    cv.namedWindow("rows"); cv.setMouseCallback("rows", cb)
    print("WINDOW_CREATED");  # should print immediately
    cv.imshow("rows", img); cv.waitKey(1)  # force the first paint

    while True:
        view = img.copy()
        # draw row prompts and collected pairs
        r = len(clicks)//2
        cv.putText(view, f"Row {r+1}/{args.rows}: click LEFT then RIGHT",
                   (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        for i,(x,y) in enumerate(clicks):
            cv.circle(view, (x,y), 6, (0,255,0), -1)
            cv.putText(view, f"{i}", (x+6,y-6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            if i%2==1:
                x0,y0 = clicks[i-1]
                cv.line(view, (x0,y0), (x,y), (0,200,200), 1)
        cv.imshow("rows", view)
        k = cv.waitKey(20)
        if k == 27:  # Esc
            raise SystemExit("Canceled")
        if k in (ord('z'), ord('Z')) and clicks:
            clicks.pop()
        if k == 13 and len(clicks) == 2*args.rows:  # Enter
            break

    # build per-row points
    xy_uv = []
    overlay = img.copy()
    for j in range(args.rows):
        lx,ly = clicks[2*j]
        rx,ry = clicks[2*j+1]
        ncols = cols_per_row[j]
        denom = max(ncols-1, 1)
        step = np.array([(rx-lx)/denom, (ry-ly)/denom], np.float32)
        base = np.array([lx,ly], np.float32)
        for i in range(ncols):
            pt = base + i*step
            u,v = to_board_norm(float(pt[0]), float(pt[1]), H)
            xy_uv.append([u,v])
            cv.circle(overlay, (int(pt[0]), int(pt[1])), 3, (0,0,255), -1)

    json.dump({"layout_id": args.layout_id, "xy": xy_uv}, open(LAYOUT_PATH, "w"))
    cv.imwrite(str(BASE / "layout_preview.png"), overlay)
    print(f"Wrote {LAYOUT_PATH} with {len(xy_uv)} LEDs. Preview: layout_preview.png")

if __name__ == "__main__":
    main()
