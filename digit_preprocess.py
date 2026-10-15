"""Same preprocessing as the browser app (app_template.html -> preprocess()).
Takes a 2-D grayscale array (stroke bright) of any size, returns 64 features in 0..1."""
import numpy as np

TARGET = 8   # longest side of the digit after resizing, inside the 8x8 grid

def preprocess_np(g):
    g = np.asarray(g, dtype=float)
    if g.max() <= 0: return None
    ys, xs = np.where(g > 0.25 * g.max())
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    s = TARGET / max(w, h)
    tw, th = max(1, round(w * s)), max(1, round(h * s))
    out = np.zeros((8, 8)); ox, oy = (8 - tw) // 2, (8 - th) // 2
    for j in range(th):
        ya = y0 + (j * h) // th; yb = y0 + max((j * h) // th + 1, -(-((j + 1) * h) // th))
        for i in range(tw):
            xa = x0 + (i * w) // tw; xb = x0 + max((i * w) // tw + 1, -(-((i + 1) * w) // tw))
            out[oy + j, ox + i] = g[ya:yb, xa:xb].mean()
    return (out / out.max()).reshape(64)

def render(img8, k, oy, ox, size=280):
    """Draw an 8x8 sklearn digit (0..1) as a big grayscale canvas, like a drawing."""
    big = np.kron(img8.reshape(8, 8) * 255, np.ones((k, k)))[:size, :size]
    c = np.zeros((size, size)); c[oy:oy + big.shape[0], ox:ox + big.shape[1]] = big
    return c
