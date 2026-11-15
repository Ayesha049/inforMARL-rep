"""Digitize the InforMARL (blue) mean curve from Figure 3 of Nayak et al. (2023).

Calibration uses the gridlines: their pixel positions are mapped to the known
tick values of each panel, by a least-squares linear fit.
"""
import json
import sys

import numpy as np
from PIL import Image

img = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
H, W, _ = img.shape
top = img[: int(H * 0.30)]  # Figure 3 sits in the top of page 7
R, G, B = top[..., 0], top[..., 1], top[..., 2]

dark = (R < 60) & (G < 60) & (B < 60)
# axes boxes: rows with long runs of dark pixels
row_counts = dark.sum(1)
cand_rows = np.where(row_counts > W * 0.15)[0]


def runs(mask_1d, min_len):
    out, start = [], None
    for i, v in enumerate(mask_1d):
        if v and start is None:
            start = i
        if (not v or i == len(mask_1d) - 1) and start is not None:
            end = i if not v else i + 1
            if end - start >= min_len:
                out.append((start, end))
            start = None
    return out


# horizontal spines: for each candidate row, the dark runs give panel x-extents
boxes = []
for r in cand_rows:
    for a, b in runs(dark[r], int(W * 0.15)):
        boxes.append((r, a, b))
# group by x-extent: top spine and bottom spine of each panel share (a, b)
panels = {}
for r, a, b in boxes:
    key = (a // 10, b // 10)
    panels.setdefault(key, []).append(r)
panel_list = []
for (ka, kb), rows in panels.items():
    rows = sorted(rows)
    if rows[-1] - rows[0] < H * 0.05:
        continue
    a = ka * 10
    xs = [bx for (r, bx, bb) in boxes if (bx // 10, bb // 10) == (ka, kb)]
    xe = [bb for (r, bx, bb) in boxes if (bx // 10, bb // 10) == (ka, kb)]
    panel_list.append((min(xs), max(xe), rows[0], rows[-1]))
panel_list.sort()
print("panels (x0,x1,y0,y1):", panel_list, file=sys.stderr)

TICKS = {  # per panel, left to right: (x ticks, y ticks)
    3: (np.arange(0, 2.01, 0.25) * 1e6, np.arange(-150, 201, 50)),
    # 7 agents: the 400 gridline coincides with the top spine, so it is not detectable
    7: (np.arange(0, 4.01, 0.5) * 1e6, np.arange(-400, 301, 100)),
    10: (np.arange(0, 5.01, 1.0) * 1e6, np.arange(-400, 401, 200)),
}


def gridlines(sub, axis):
    """Positions of light-gray gridlines inside a panel (axis=0: vertical)."""
    r, g, b = sub[..., 0], sub[..., 1], sub[..., 2]
    gray = (abs(r - g) < 8) & (abs(g - b) < 8) & (r > 150) & (r < 215)
    frac = gray.mean(0 if axis == 0 else 1)
    idx = np.where(frac > 0.25)[0]
    # collapse adjacent indices
    groups, cur = [], [idx[0]] if len(idx) else []
    for i in idx[1:]:
        if i - cur[-1] <= 2:
            cur.append(i)
        else:
            groups.append(np.mean(cur)); cur = [i]
    if cur:
        groups.append(np.mean(cur))
    return np.array(groups)


out = {}
for n, (x0, x1, y0, y1) in zip((3, 7, 10), panel_list):
    pad = 4
    sub = top[y0 + pad : y1 - pad, x0 + pad : x1 - pad]
    gx = gridlines(sub, 0) + x0 + pad
    gy = gridlines(sub, 1) + y0 + pad
    xt, yt = TICKS[n]
    print(f"n={n}: {len(gx)} vertical gridlines (expect {len(xt)}), "
          f"{len(gy)} horizontal (expect {len(yt)})", file=sys.stderr)
    assert len(gx) == len(xt) and len(gy) == len(yt), "gridline detection failed"
    ax_fit = np.polyfit(gx, xt, 1)
    ay_fit = np.polyfit(gy, yt[::-1], 1)  # pixel y grows downward
    res_x = np.abs(np.polyval(ax_fit, gx) - xt).max()
    res_y = np.abs(np.polyval(ay_fit, gy) - yt[::-1]).max()
    print(f"   calibration residuals: x {res_x:.0f} steps, y {res_y:.2f} reward", file=sys.stderr)
    # matplotlib default blue #1f77b4 = (31, 119, 180)
    blk = top[y0 + pad : y1 - pad, x0 + pad : x1 - pad]
    d = np.sqrt(((blk - np.array([31, 119, 180])) ** 2).sum(-1))
    blue = d < 45
    pts = []
    for c in range(blue.shape[1]):
        ys = np.where(blue[:, c])[0]
        if len(ys) == 0:
            continue
        py = np.median(ys) + y0 + pad
        px = c + x0 + pad
        pts.append((float(np.polyval(ax_fit, px)), float(np.polyval(ay_fit, py))))
    out[n] = pts
    print(f"   {len(pts)} curve columns, first {pts[0]}, last {pts[-1]}", file=sys.stderr)

json.dump(out, open(sys.argv[2], "w"))
