#!/usr/bin/env python3
"""
ASCII Candle
============
A match is struck, it lights a candle, the flame grows, the candle tips over
and the whole floor goes up in flames - all drawn with coloured ASCII
characters in a tkinter window.

Run:     python candle.py
Keys:    R = restart   Space = pause   Esc = quit
Export:  python candle.py --export candle.gif      (needs Pillow)
"""
import argparse
import math
import random
import sys
import time

import numpy as np

# ---------------------------------------------------------------- settings --
COLS, ROWS = 100, 62          # character grid
CW, CH = 7.0, 12.0            # size of one cell in "world" pixels
W, H = COLS * CW, ROWS * CH   # world size
FPS = 30

# dim -> bright character ramp
RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
N = len(RAMP) - 1
ALL = RAMP + "="              # every char we can draw (ramp + floor pattern)
CI = {c: i for i, c in enumerate(ALL)}

# timeline (seconds)
T_STRIKE = 1.2                # match is struck
T_LIGHT = 3.2                 # match flame touches the wick
T_OUT = 3.9                   # match is gone
T_TIP = 8.6                   # candle starts tipping over
T_FALL = 0.55                 # how long the fall takes
T_IGNITE = T_TIP + T_FALL     # flame hits the floor

# scene geometry
FLOOR_Y = H * 0.80
CAND_W = 13 * CW              # candle width
CAND_H = 20 * CH              # candle height
CAND_X = W * 0.40             # candle centre x
PIVOT = (CAND_X + CAND_W / 2, FLOOR_Y)   # bottom-right corner, it falls right

# colours
WAX = np.array((0.78, 0.58, 0.32))
BURNT = np.array((0.07, 0.05, 0.04))
EMBER = np.array((0.62, 0.12, 0.05))
ORANGE = np.array((1.0, 0.55, 0.15))
WHITE = np.array((1.0, 0.95, 0.78))
BG_COL = np.array((0.60, 0.25, 0.09))
FLOOR_COL = np.array((0.80, 0.42, 0.18))

# grid of cell centres
XS = (np.arange(COLS) + 0.5) * CW
YS = (np.arange(ROWS) + 0.5) * CH
X, Y = np.meshgrid(XS, YS)
_rng = np.random.default_rng(7)
JIT = _rng.uniform(-1, 1, (ROWS, COLS))            # static per-cell jitter


# ----------------------------------------------------------------- helpers --
def clamp(v, a=0.0, b=1.0):
    return np.clip(v, a, b) if isinstance(v, np.ndarray) else max(a, min(b, v))


def smoothstep(a, b, x):
    u = clamp((x - a) / (b - a))
    return u * u * (3 - 2 * u)


def mix(a, b, k):
    """Linear blend of colours a and b; k may be a scalar or an array."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    k = np.asarray(k, float)
    return a + (b - a) * k[..., None] if k.ndim else a + (b - a) * k


def _hash(ix, iy):
    h = np.sin(ix * 127.1 + iy * 311.7) * 43758.5453
    return h - np.floor(h)


def nz2(x, y):
    """Smooth 2-D value noise in [0, 1]."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash(ix, iy), _hash(ix + 1, iy)
    c, d = _hash(ix, iy + 1), _hash(ix + 1, iy + 1)
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy


def nz(x, seed=0.0):
    return float(nz2(x, seed))


def fbm(x, y):
    return (0.5 * nz2(x, y) + 0.3 * nz2(x * 2.1 + 5.2, y * 2.1 + 1.3)
            + 0.2 * nz2(x * 4.3 + 9.1, y * 4.3 + 7.7))


def ramp_index(lum, jit, spread=2.5):
    """Map luminance to an index in RAMP (0 = blank)."""
    idx = clamp(lum, 0, 1) ** 0.7 * N + jit * spread
    idx = np.clip(idx, 3, N).astype(np.int16)
    idx[lum < 0.03] = 0
    return idx


def shade(col, lum):
    """Colour of a glyph with base colour `col` and brightness `lum`."""
    lum = np.asarray(lum, float)
    rgb = col * (0.22 + 0.85 * clamp(lum, 0, 1.0))[..., None]
    hot = clamp(lum - 0.85, 0, 1)[..., None] * 2.0
    return np.clip(rgb + (WHITE - rgb) * np.clip(hot, 0, 1), 0, 1)


def flame_field(fx, fy, fh, fw, sway, t, seed):
    """Tear-drop flame anchored at (fx, fy). Returns (intensity, core)."""
    if fh <= 1:
        z = np.zeros_like(X)
        return z, z
    v = (fy - Y) / fh                                   # 0 at base, 1 at tip
    dx = X - (fx + sway * clamp(v, 0, 1) ** 2)
    half = fw / 2
    turb = (fbm(X * 0.09 + seed, Y * 0.09 + t * 7.0) - 0.5) * 0.55
    r_low = np.hypot(dx / half, (v - 0.25) / 0.40)
    taper = np.maximum(1e-3, half * clamp(1 - (v - 0.25) / 0.75) ** 0.9)
    r_up = np.abs(dx) / taper
    r = np.where(v < 0.25, r_low, r_up) + turb
    r = np.where((v > 1.0) | (v < -0.2), 9.0, r)
    inten = clamp(1.15 - r)
    core = clamp(1.0 - r * 1.7) * clamp(1 - v * 0.7)
    return inten, core


# ------------------------------------------------------------------- scene --
class Scene:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.parts = []          # [x, y, vx, vy, life, max_life, kind]
        self.sparked = False
        self.ember_acc = 0.0
        self.prev_theta = 0.0

    # --- state of each actor at time t ---------------------------------
    @staticmethod
    def candle_angle(t):
        u = clamp((t - T_TIP) / T_FALL)
        th = (math.pi / 2) * u * u
        if t > T_IGNITE:                                   # small bounce
            tau = t - T_IGNITE
            th = math.pi / 2 - 0.12 * math.exp(-6 * tau) * abs(math.sin(tau * 12))
        return th

    @staticmethod
    def to_world(lx, ly, th):
        c, s = math.cos(th), math.sin(th)
        return PIVOT[0] + c * lx - s * ly, PIVOT[1] + s * lx + c * ly

    @staticmethod
    def match_head(t):
        wick = (CAND_X - CAND_W * 0.05, FLOOR_Y - CAND_H + 6)
        keys = [(T_STRIKE, (W * 0.07, H * 0.62)),
                (T_STRIKE + 0.9, (W * 0.15, H * 0.53)),
                (T_LIGHT, wick),
                (T_OUT, (W * 0.16, H * 0.58))]
        if t <= keys[0][0]:
            return keys[0][1]
        for (ta, pa), (tb, pb) in zip(keys, keys[1:]):
            if t <= tb:
                k = smoothstep(ta, tb, t)
                return pa[0] + (pb[0] - pa[0]) * k, pa[1] + (pb[1] - pa[1]) * k
        return keys[-1][1]

    # --- particles --------------------------------------------------------
    def spawn(self, x, y, vx, vy, life, kind):
        self.parts.append([x, y, vx, vy, life, life, kind])

    def update_particles(self, t, dt):
        alive = []
        for p in self.parts:
            p[4] -= dt
            if p[4] <= 0:
                continue
            if p[6] == "spark":
                p[3] += 420 * dt                           # gravity
            else:                                          # ember: rises, drifts
                p[2] += (nz(p[1] * 0.02 + t, p[0] * 0.01) - 0.5) * 240 * dt
                p[3] -= 25 * dt
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            alive.append(p)
        self.parts = alive

    # --- render one frame -------------------------------------------------
    def frame(self, t, dt):
        """Return (char_index[ROWS, COLS], rgb[ROWS, COLS, 3]) for time t."""
        lights = []                                        # (x, y, strength, radius)
        flames = []                                        # (intensity, core)

        # ---- candle state
        th = self.candle_angle(t)
        omega = (th - self.prev_theta) / max(dt, 1e-3)
        self.prev_theta = th
        lit = smoothstep(T_LIGHT, T_LIGHT + 0.25, t)
        grow = smoothstep(T_LIGHT, T_LIGHT + 1.6, t)
        fs = lit * (0.35 + 0.65 * grow) * (1 + 0.45 * smoothstep(T_LIGHT + 2, T_TIP, t))
        hcur = CAND_H - 14 * clamp((t - T_LIGHT) / (T_TIP - T_LIGHT))
        wx, wy = self.to_world(-CAND_W / 2, -hcur - 4, th)  # wick tip

        if fs > 0:
            flick = 0.85 + 0.3 * nz(t * 6.0, 1.7)
            fh = 105 * fs * flick
            fw = 58 * fs * (0.9 + 0.2 * nz(t * 4.0, 4.2))
            sway = (nz(t * 2.5, 9.1) - 0.5) * 22 * fs - omega * 14
            flames.append(flame_field(wx, wy + 6, fh, fw, sway, t, 0.0))
            lights.append((wx, wy - fh * 0.35, 1.0 * fs * flick, 170))

        # ---- match
        m_on = smoothstep(T_STRIKE, T_STRIKE + 0.15, t) * (1 - smoothstep(T_OUT - 0.35, T_OUT, t))
        mx, my = self.match_head(t)
        if t >= T_STRIKE and not self.sparked:
            self.sparked = True
            for _ in range(34):
                a = self.rng.uniform(-math.pi, 0.2)
                sp = self.rng.uniform(60, 260)
                self.spawn(mx, my, math.cos(a) * sp + 80, math.sin(a) * sp,
                           self.rng.uniform(0.25, 0.7), "spark")
        if m_on > 0.01:
            flare = 1 + 0.8 * math.exp(-(t - T_STRIKE) * 6)
            mf = 0.9 + 0.2 * nz(t * 9, 3.3)
            flames.append(flame_field(mx, my - 2, 62 * flare * m_on * mf,
                                      28 * flare, (nz(t * 3, 5.5) - 0.5) * 8, t, 11.0))
            lights.append((mx, my - 15, 0.55 * m_on, 110))

        # ---- floor fire
        fire = np.zeros_like(X)
        fire_core = np.zeros_like(X)
        band = np.zeros(COLS)
        col_a = np.zeros(COLS)
        if t > T_IGNITE:
            age = t - T_IGNITE
            ix = wx
            speed = 230 + 170 * age
            reach = 230 * age + 85 * age * age
            col_age = (reach - np.abs(XS - ix)) / speed
            col_a = clamp(col_age / 0.7)
            lay_cx = PIVOT[0] + hcur * 0.5
            hcol = col_a * ((55 + 120 * nz2(XS * 0.011 + 3.1, t * 0.35))
                            * (0.55 + 0.9 * nz2(XS * 0.045, t * 2.2))
                            + 90 * np.exp(-((XS - lay_cx) / 110) ** 2))
            d = FLOOR_Y - Y
            n = fbm(X * 0.05, Y * 0.07 + t * 4.5)
            n = clamp((n - 0.5) * 2.4 + 0.5)
            hh = hcol[None, :] * (0.35 + 1.1 * n) + 1e-3
            fire = clamp(1.05 - d / hh) * (d > -CH * 0.5) * (hcol[None, :] > 1)
            fire_core = clamp(fire * 1.6 - 0.7) * clamp(1 - (FLOOR_Y - Y) / 60)
            band = col_a * (0.75 + 0.25 * nz2(XS * 0.08, t * 3.0))
            # lights along the burning stretch
            lo, hi = max(0.0, ix - reach), min(W, ix + reach)
            if hi > lo:
                for k in range(10):
                    lx_ = lo + (hi - lo) * (k + 0.5) / 10
                    ci_ = int(clamp(lx_ / CW, 0, COLS - 1))
                    lights.append((lx_, FLOOR_Y - hcol[ci_] * 0.45,
                                   0.75 * col_a[ci_], 210))

        # ---- light field (sum of soft lights)
        LF = np.zeros_like(X)
        for (lx_, ly_, s_, r_) in lights:
            dd = np.hypot(X - lx_, Y - ly_)
            LF += s_ * (np.exp(-(dd / r_) ** 2) + 0.12 * np.exp(-dd / (1.5 * r_)))

        # ---- layer 1: background (only what the light reaches)
        lum = LF * 0.06
        ci = np.clip(lum / 0.12 * 7 + JIT * 1.5, 1, 8).astype(np.int16)
        ci[lum < 0.022] = 0
        rgb = shade(BG_COL, lum * 2.5)

        # ---- layer 2: floor
        floor = Y >= FLOOR_Y
        fr = np.floor((Y - FLOOR_Y) / CH).astype(int)
        cidx = np.arange(COLS)[None, :].repeat(ROWS, 0)
        pat = np.where(fr == 0,
                       np.where(cidx % 3 == 0, CI["+"], CI["="]),
                       np.where((cidx + fr * 2) % 6 == 0, CI["+"],
                                np.where((cidx + fr) % 2 == 0, CI["-"], 0)))
        f_lum = LF * 0.6 * np.exp(-np.maximum(fr, 0) * 0.22)
        f_lum = np.where(fr == 0, f_lum * 1.3, f_lum)
        m = floor & (f_lum > 0.04) & (pat > 0)
        ci[m] = pat[m]
        rgb[m] = shade(FLOOR_COL, f_lum)[m]
        ci[floor & ~m] = 0

        # ---- layer 3: candle body
        dx, dy = X - PIVOT[0], Y - PIVOT[1]
        c, s = math.cos(th), math.sin(th)
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        inside = (lx >= -CAND_W) & (lx <= 0) & (ly >= -hcur) & (ly <= 0)
        if inside.any():
            across = np.abs((lx + CAND_W / 2) / (CAND_W / 2))
            depth = hcur + ly                               # px below the top
            gm = lit * (0.8 + 0.2 * nz(t * 8, 2.2))
            glow = gm * np.exp(-depth / 36.0)
            burnt = clamp((t - T_IGNITE) / 5.0) * 0.8
            k = (0.05 + glow * 1.25 + LF * 0.55) * (1 - 0.45 * across) * (1 - 0.45 * burnt)
            col = mix(WAX, ORANGE, clamp(glow))
            col = col + (BURNT - col) * burnt
            c_ci = ramp_index(k, JIT, 1.5)
            c_rgb = shade(col, k)
            ci[inside] = c_ci[inside]
            rgb[inside] = c_rgb[inside]
            # dark wick while the candle is unlit
            if lit < 0.5 and th < 0.3:
                r_, c_ = int(wy // CH), int(wx // CW)
                if 0 <= r_ < ROWS and 0 <= c_ < COLS:
                    ci[r_, c_] = CI["|"]
                    rgb[r_, c_] = (0.30, 0.24, 0.20)

        # ---- layer 4: burning floor band (white-hot line under the fire)
        if t > T_IGNITE:
            for k_, gain in enumerate((1.15, 0.85, 0.5, 0.28)):
                row = int((FLOOR_Y + k_ * CH) // CH)
                if row >= ROWS:
                    break
                glow = band * gain
                on = glow > 0.12
                if k_ == 0:
                    chars = np.where(glow > 0.3, CI["@"], CI["0"])
                else:
                    chars = ramp_index(glow[None, :], JIT[row:row + 1], 3.0)[0]
                col = mix(EMBER, ORANGE, clamp(glow))
                rgb_row = shade(col, glow * 1.1)
                ci[row, on] = chars[on]
                rgb[row, on] = rgb_row[on]

        # ---- layer 5: flames (emissive, drawn over everything)
        djit = (nz2(X * 0.9, Y * 0.9 + t * 9.0) - 0.5) * 2.2
        flames.append((fire, fire_core))
        for inten, core in flames:
            fl = inten * 0.8 + core * 0.35
            m = fl > 0.06
            if not m.any():
                continue
            m &= fl >= 0.6 * clamp(rgb.max(axis=2) - 0.25, 0, 1)
            col = mix(EMBER, ORANGE, clamp(inten * 1.4 - 0.1))
            col = col + (WHITE - col) * clamp(core)[..., None]
            ci[m] = ramp_index(fl, djit, 3.0)[m]
            rgb[m] = shade(col, 0.35 + fl)[m]

        # ---- layer 6: particles (sparks + embers)
        if t > T_IGNITE:
            self.ember_acc += dt * 55 * float(col_a.mean() * 2 + 0.2)
            while self.ember_acc > 1:
                self.ember_acc -= 1
                c_ = self.rng.randrange(COLS)
                if col_a[c_] > 0.3:
                    self.spawn(XS[c_], FLOOR_Y - self.rng.uniform(20, 140),
                               self.rng.uniform(-20, 20), self.rng.uniform(-150, -60),
                               self.rng.uniform(1.0, 3.2), "ember")
        self.update_particles(t, dt)
        for x_, y_, _, _, life, ml, kind in self.parts:
            r_, c_ = int(y_ // CH), int(x_ // CW)
            if not (0 <= r_ < ROWS and 0 <= c_ < COLS):
                continue
            k = life / ml
            ci[r_, c_] = CI["*" if k > 0.6 else "+" if k > 0.3 else "."]
            rgb[r_, c_] = mix(ORANGE, WHITE, k) * (0.45 + 0.55 * k)

        return ci, rgb


# ------------------------------------------------------------- tk front end --
def encode(ci, rgb):
    """Pack char + 5-bit colour into one int per cell (for cheap diffing)."""
    q = (np.clip(rgb, 0, 1) * 31 + 0.5).astype(np.int32)
    key = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    key[ci == 0] = 0
    return (key * len(ALL) + ci).ravel(), key.ravel(), ci.ravel()


COLORS = ["#%02x%02x%02x" % (((k >> 10) & 31) * 255 // 31, ((k >> 5) & 31) * 255 // 31,
                             (k & 31) * 255 // 31) for k in range(1 << 15)]


def pick_font(tkfont, root):
    families = set(tkfont.families(root))
    for name in ("Consolas", "Cascadia Mono", "Menlo", "DejaVu Sans Mono",
                 "Liberation Mono", "Courier New", "Courier"):
        if name in families:
            return tkfont.Font(root=root, family=name, size=9, weight="bold")
    return tkfont.Font(root=root, family="TkFixedFont", size=9, weight="bold")


def run_gui():
    import tkinter as tk
    from tkinter import font as tkfont

    root = tk.Tk()
    root.title("Candle")
    root.configure(bg="black")
    fnt = pick_font(tkfont, root)
    cw = max(1, fnt.measure("M"))
    ch = max(1, fnt.metrics("linespace"))
    canvas = tk.Canvas(root, width=COLS * cw, height=ROWS * ch, bg="black",
                       highlightthickness=0)
    canvas.pack()
    items = [canvas.create_text(c * cw, r * ch, text=" ", fill="#000000",
                                font=fnt, anchor="nw")
             for r in range(ROWS) for c in range(COLS)]
    call, path = canvas.tk.call, canvas._w

    st = {"scene": Scene(), "t": 0.0, "last": time.perf_counter(),
          "paused": False, "prev": np.full(ROWS * COLS, -1, np.int64)}

    def restart(_=None):
        st["scene"] = Scene()
        st["t"] = 0.0
        st["last"] = time.perf_counter()

    def toggle(_=None):
        st["paused"] = not st["paused"]
        st["last"] = time.perf_counter()

    root.bind("<r>", restart)
    root.bind("<R>", restart)
    root.bind("<space>", toggle)
    root.bind("<Escape>", lambda e: root.destroy())

    def tick():
        t0 = time.perf_counter()
        dt = min(0.1, t0 - st["last"])
        st["last"] = t0
        if not st["paused"]:
            st["t"] += dt
            ci, rgb = st["scene"].frame(st["t"], dt)
            codes, keys, chars = encode(ci, rgb)
            for i in np.nonzero(codes != st["prev"])[0]:
                call(path, "itemconfigure", items[i], "-text", ALL[chars[i]],
                     "-fill", COLORS[keys[i]])
            st["prev"] = codes
        spent = (time.perf_counter() - t0) * 1000
        root.after(max(1, int(1000 / FPS - spent)), tick)

    tick()
    root.mainloop()


# ---------------------------------------------------------------- exporter --
def export(path, seconds, fps):
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for f in ("consolab.ttf", "DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            font = ImageFont.truetype(f, 11)
            break
        except OSError:
            pass
    font = font or ImageFont.load_default()
    cw, ch = 7, 12
    scene, frames = Scene(seed=1), []
    total = int(seconds * fps)
    for f in range(total):
        t = f / fps
        ci, rgb = scene.frame(t, 1 / fps)
        img = Image.new("RGB", (COLS * cw, ROWS * ch), "black")
        d = ImageDraw.Draw(img)
        col8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        for r, c in zip(*np.nonzero(ci)):
            d.text((c * cw, r * ch), ALL[ci[r, c]], fill=tuple(int(v) for v in col8[r, c]), font=font)
        frames.append(img)
        print(f"\rframe {f + 1}/{total}", end="", flush=True)
    print()
    if path.lower().endswith(".gif"):
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)
    else:
        import os
        os.makedirs(path, exist_ok=True)
        for i, im in enumerate(frames):
            im.save(os.path.join(path, f"frame_{i:04d}.png"))
    print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ASCII candle animation")
    ap.add_argument("--export", metavar="OUT", help="render to a .gif (or a folder of PNGs) instead of a window")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()
    if a.export:
        export(a.export, a.seconds, a.fps)
    else:
        run_gui()
