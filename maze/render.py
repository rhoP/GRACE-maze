"""Top-down images of the maze (numpy raster) and episode GIFs."""

from __future__ import annotations

import numpy as np

from maze import game as G
from maze import levels as L
from maze import sim

COL = {"floor": (236, 232, 222), "wall": (60, 60, 70), "door": (176, 120, 70), "flap": (200, 150, 40),
       "pit": (40, 30, 30), "key": (60, 140, 220), "agent": (40, 170, 90), "dead": (200, 40, 60),
       "goal": [(212, 175, 55), (150, 150, 165), (140, 80, 190)], "trail": (120, 200, 140)}
PX = 0.05  # units per pixel


def _grid(spec):
    n = int(sim.GM * sim.R / PX)
    c = (np.arange(n) + 0.5) * PX
    return np.meshgrid(c, c)


def frame(spec, st, trail=None, crop: bool = True) -> np.ndarray:
    """RGB image of a game state (y up), cropped to the level's rooms."""
    px, py = _grid(spec)
    img = np.zeros(px.shape + (3,), np.uint8)
    img[:] = COL["floor"]
    x1, y1 = spec.gw * sim.R, spec.gh * sim.R
    img[(px > x1) | (py > y1)] = COL["wall"]
    for r in spec.pits:
        img[L.rect_dist(px, py, r) <= 0] = COL["pit"]
    kind, idx = L.edge_kinds(spec)
    doors, flaps, gone = int(st.doors), int(st.flaps), int(st.gone)
    for r in L.static_rects():
        img[L.rect_dist(px, py, r) <= 0] = COL["wall"]
    for k in range(sim.NEDGE):
        if kind[k] == 0 or (kind[k] == 3 and (flaps >> idx[k]) & 1):
            img[L.rect_dist(px, py, L.filler_rect(k)) <= 0] = COL["wall"]
        elif kind[k] == 2 and not (doors >> idx[k]) & 1:
            img[L.rect_dist(px, py, L.filler_rect(k)) <= 0] = COL["door"]
    for lk in spec.flaps:
        if (flaps >> lk.flap) & 1:
            continue
        to_b = spec.frame_room(lk.to) == sim.EDGES[lk.edge][1]
        pos, quat = sim.flap_pose(lk.edge, to_b)
        ang = 2 * np.arctan2(quat[3], quat[0]) + float(np.asarray(st.qpos)[2 + lk.flap])
        d = np.array([np.cos(ang), np.sin(ang)])
        s = np.clip(((px - pos[0]) * d[0] + (py - pos[1]) * d[1]) / sim.GAP, 0, 1)
        dist = np.hypot(px - pos[0] - s * sim.GAP * d[0], py - pos[1] - s * sim.GAP * d[1])
        img[dist < 0.06] = COL["flap"]
    for i, (x, y, k, _) in enumerate(spec.items):
        if not (gone >> i) & 1:
            img[np.hypot(px - x, py - y) < 0.22] = COL["key"] if k == G.KEY else COL["goal"][k]
    if trail is not None:
        for q in trail:
            img[np.hypot(px - q[0], py - q[1]) < 0.05] = COL["trail"]
    ax, ay = np.asarray(st.qpos)[:2]
    img[np.hypot(px - ax, py - ay) < sim.RAD] = COL["agent"] if bool(st.alive) else COL["dead"]
    img = img[::-1]
    if crop:
        n, h, w = img.shape[0], int(round(spec.gh * sim.R / PX)), int(round(spec.gw * sim.R / PX))
        img = img[n - h:, :w]
    return img


def level_map(spec, ax, title: str = ""):
    """The level with its room graph: open links solid, doors dashed, flaps as arrows."""
    st = G.State(np.r_[spec.spawn, np.zeros(sim.NF)], np.zeros(2 + sim.NF), 0, spec.spawn_room, spec.keys0, 0, 0, 0,
                 -1, True)
    H = sim.GM * sim.R
    ax.imshow(frame(spec, st, crop=False), extent=(0, H, 0, H), interpolation="nearest")
    cen = lambda r: np.array([(r % spec.gw + 0.5) * sim.R, (r // spec.gw + 0.5) * sim.R])  # noqa: E731
    for lk in spec.links:
        a, b = cen(lk.a), cen(lk.b)
        if lk.kind == "flap":
            f, t = cen(lk.frm), cen(lk.to)
            ax.annotate("", xy=f + 0.7 * (t - f), xytext=f + 0.3 * (t - f),
                        arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=1.8))
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="k" if lk.kind == "open" else "saddlebrown", lw=1.4,
                    ls="-" if lk.kind == "open" else "--", alpha=0.6)
    for i, (x, y, k, _) in enumerate(spec.items):
        if k != G.KEY:
            ax.text(x, y + 0.3, "ABC"[k], ha="center", va="bottom", fontsize=8, weight="bold")
    ax.set_xlim(0, spec.gw * sim.R)
    ax.set_ylim(0, spec.gh * sim.R)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def gif(spec, states, path: str, fps: int = 20, every: int = 1):
    import imageio.v2 as imageio
    trail, frames = [], []
    for i, st in enumerate(states):
        trail.append(np.asarray(st.qpos)[:2])
        if i % every == 0 or i == len(states) - 1:
            frames.append(frame(spec, st, trail))
    imageio.mimsave(path, frames, duration=1 / fps, loop=0)


__all__ = ["frame", "level_map", "gif"]
