"""Level family: rooms on a grid joined by open gaps, key doors or one-way flaps, with goals, keys and pits.

A level is GH × GW rooms of the frame (``sim``), joined by a random spanning tree plus 1–2 extra links. A link is
``open``, a ``door`` (needs a key, which is spent; at most ND) or a ``flap`` (one-way from ``frm`` to ``to``, and
latched after one crossing; at most NF). Every other edge is a wall.

Each room has 0–2 pits (rectangles) kept off the approaches to its gaps and off its centre. Items: goals A, B, C in
three rooms other than the spawn room, 0–2 key pickups, and 1 key at the start. A level is accepted when ≥ 2 goals are
reachable, some transition shrinks the face, A is strictly the cheapest (fewest key uses and pickups before it), and
every room's gaps, items and spawn are connected on the planner grid. Goals are relabelled by cost and stored at
item indices 0–2 (A, B, C); keys follow. Payoffs (1, 0.6, 0.3) go by rank; ``mpref`` uses the rankings with A last.
"""

from __future__ import annotations

import itertools
import os
import pickle
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from maze import game as G
from maze import sim

VALS = (1.0, 0.6, 0.3)
RANKS = list(itertools.permutations(range(3)))
TRAIN_RANKS = [r for r in RANKS if r[0] != 2]
PREF_RANKS = [r for r in RANKS if r[0] == 2]
SPLITS = {"mtrain": dict(g=(3, 3), ranks="train", seed=320), "mtest": dict(g=(3, 3), ranks="train", seed=620),
          "mpref": dict(g=(3, 3), ranks="pref", seed=520), "mbig": dict(g=(4, 4), ranks="train", seed=720)}
CACHE = os.environ.get("MAZE_LEVEL_CACHE", "runs/level_cache")
RES = 0.1  # planner grid
GRES = 0.25  # observation raster
GPAD = 8
GP = int(sim.GM * sim.R / GRES) + 2 * GPAD


def payoff(rank) -> np.ndarray:
    return np.array([VALS[r] for r in rank], np.float32)


@dataclass
class Link:
    a: int
    b: int
    kind: str  # "open", "door" or "flap"
    edge: int  # frame edge
    door: int = -1
    flap: int = -1
    frm: int = -1  # flap: passable from this room
    to: int = -1

    def other(self, r: int) -> int:
        return self.b if r == self.a else self.a


@dataclass
class Spec:
    gh: int
    gw: int
    links: list
    items: list  # (x, y, kind, room); goals at 0-2 with kind = index, keys after
    pits: list  # (x0, y0, x1, y1)
    spawn: tuple
    spawn_room: int
    keys0: int
    seed: int

    @property
    def n_rooms(self) -> int:
        return self.gh * self.gw

    @property
    def doors(self) -> list:
        return [lk for lk in self.links if lk.kind == "door"]

    @property
    def flaps(self) -> list:
        return [lk for lk in self.links if lk.kind == "flap"]

    def frame_room(self, r: int) -> int:
        i, j = divmod(r, self.gw)
        return i * sim.GM + j

    def room_box(self, r: int):
        i, j = divmod(r, self.gw)
        return j * sim.R, i * sim.R, (j + 1) * sim.R, (i + 1) * sim.R

    def room_edges(self, r: int) -> list:
        """(frame edge, neighbour level room or -1, normal out of r) for the 4 sides of room r."""
        i, j = divmod(r, self.gw)
        out = []
        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ii, jj = i + di, j + dj
            fr = self.frame_room(r)
            if 0 <= ii < sim.GM and 0 <= jj < sim.GM:
                k = sim.EDGE_OF[(fr, ii * sim.GM + jj)]
                nb = ii * self.gw + jj if (0 <= ii < self.gh and 0 <= jj < self.gw) else -1
                out.append((k, nb, np.array([dj, di], float)))
        return out


def approach(spec: Spec, lk: Link, room: int, d: float = 0.6) -> np.ndarray:
    """A point d inside ``room`` in front of the link's gap."""
    c = sim.gap_centre(lk.edge)
    n = sim.normal(lk.edge, to_b=(spec.frame_room(room) == sim.EDGES[lk.edge][1]))
    return c + n * d


# ------------------------------------------------------------------ geometry for planning and rasters

@lru_cache(maxsize=None)
def static_rects() -> np.ndarray:
    m, _, _, ids = sim.model()
    out = []
    for gi in range(m.ngeom):
        name = m.geom(gi).name
        if name.startswith("bd") or (name.startswith("s") and "_" in name):
            (x, y, _), (a, b, _) = m.geom_pos[gi], m.geom_size[gi]
            out.append((x - a, y - b, x + a, y + b))
    return np.array(out)


def filler_rect(k: int) -> np.ndarray:
    c = sim.gap_centre(k)
    hx, hy = (sim.T, sim.GAP / 2 + 0.02) if sim.EDGES[k][2] == "h" else (sim.GAP / 2 + 0.02, sim.T)
    return np.array([c[0] - hx, c[1] - hy, c[0] + hx, c[1] + hy])


def edge_kinds(spec: Spec) -> tuple[np.ndarray, np.ndarray]:
    kind = np.zeros(sim.NEDGE, np.int32)
    idx = np.full(sim.NEDGE, -1, np.int32)
    for lk in spec.links:
        kind[lk.edge] = {"open": 1, "door": 2, "flap": 3}[lk.kind]
        idx[lk.edge] = lk.door if lk.kind == "door" else lk.flap
    return kind, idx


def rect_dist(px, py, r) -> np.ndarray:
    dx = np.maximum(np.maximum(r[0] - px, 0), px - r[2])
    dy = np.maximum(np.maximum(r[1] - py, 0), py - r[3])
    return np.hypot(dx, dy)


def free_grid(spec: Spec, box, doors: int = 0, flaps: int = 0, avoid=(), wall_margin: float = 0.08,
              pit_margin: float = 0.22, item_margin: float = 0.15):
    """Planner cells (centres) inside ``box`` = (x0, y0, x1, y1) that the agent's centre may occupy."""
    xs = np.arange(box[0] + RES / 2, box[2], RES)
    ys = np.arange(box[1] + RES / 2, box[3], RES)
    px, py = np.meshgrid(xs, ys)  # [iy, ix]
    free = np.ones(px.shape, bool)
    kind, idx = edge_kinds(spec)
    rects = list(static_rects())
    for k in range(sim.NEDGE):
        if kind[k] == 0 or (kind[k] == 2 and not (doors >> idx[k]) & 1) or (kind[k] == 3 and (flaps >> idx[k]) & 1):
            rects.append(filler_rect(k))
    for r in rects:
        if r[2] < box[0] - 1 or r[0] > box[2] + 1 or r[3] < box[1] - 1 or r[1] > box[3] + 1:
            continue
        free &= rect_dist(px, py, r) > sim.RAD + wall_margin
    for r in spec.pits:
        free &= rect_dist(px, py, r) > pit_margin
    for i in avoid:
        x, y = spec.items[i][:2]
        free &= np.hypot(px - x, py - y) > G.PICK + item_margin
    return free, xs, ys


def bfs_path(free, xs, ys, start, goal, start_tol: float = 0.35) -> list | None:
    """Shortest 8-connected path of cell centres from the free cell nearest ``start`` (within ``start_tol``) to the
    one nearest ``goal`` (within 0.35)."""
    fy, fx = np.nonzero(free)
    if len(fy) == 0:
        return None

    def nearest(p):
        d = (xs[fx] - p[0]) ** 2 + (ys[fy] - p[1]) ** 2
        k = int(np.argmin(d))
        return (int(fy[k]), int(fx[k])), float(np.sqrt(d[k]))
    s, ds = nearest(start)
    g, dg = nearest(goal)
    if ds > start_tol or dg > 0.35:
        return None
    H, W = free.shape
    prev = {s: None}
    q = deque([s])
    while q:
        c = q.popleft()
        if c == g:
            break
        y, x = c
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            n = (y + dy, x + dx)
            if 0 <= n[0] < H and 0 <= n[1] < W and free[n] and n not in prev:
                if dy and dx and not (free[y + dy, x] and free[y, x + dx]):
                    continue
                prev[n] = c
                q.append(n)
    if g not in prev:
        return None
    out, c = [], g
    while c is not None:
        out.append((xs[c[1]], ys[c[0]]))
        c = prev[c]
    return out[::-1]


def room_region(spec: Spec, room: int, lk: Link | None = None) -> tuple:
    """The planning box: the room, 1 unit wider on every side (gaps and the start just past one)."""
    x0, y0, x1, y1 = spec.room_box(room)
    return (x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0)


def region_mask(spec: Spec, room: int, lk: Link | None, xs, ys, start=None) -> np.ndarray:
    """Cells of the room's interior, a corridor through ``lk``'s gap, and a disc of 0.8 around ``start``."""
    px, py = np.meshgrid(xs, ys)
    x0, y0, x1, y1 = spec.room_box(room)
    m = (px > x0) & (px < x1) & (py > y0) & (py < y1)
    if start is not None:
        m |= np.hypot(px - start[0], py - start[1]) < 0.8
    if lk is not None:
        c = sim.gap_centre(lk.edge)
        if sim.EDGES[lk.edge][2] == "h":
            m |= (np.abs(py - c[1]) < sim.GAP / 2) & (np.abs(px - c[0]) < 1.0)
        else:
            m |= (np.abs(px - c[0]) < sim.GAP / 2) & (np.abs(py - c[1]) < 1.0)
    return m


# ------------------------------------------------------------------ generation

class _Reject(Exception):
    pass


def _links(rng, gh: int, gw: int, spec_stub: Spec) -> list:
    pairs = [(r, r + 1) for r in range(gh * gw) if r % gw < gw - 1] + [(r, r + gw) for r in range(gh * gw - gw)]
    parent = list(range(gh * gw))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    chosen, rest = [], []
    for k in rng.permutation(len(pairs)):
        a, b = pairs[k]
        if find(a) != find(b):
            parent[find(a)] = find(b)
            chosen.append(pairs[k])
        else:
            rest.append(pairs[k])
    chosen += rest[:int(rng.integers(1, 3))]
    links, nd, nf = [], 0, 0
    for a, b in chosen:
        e = sim.EDGE_OF[(spec_stub.frame_room(a), spec_stub.frame_room(b))]
        u = rng.random()
        if u < 0.35 and nd < G.ND:
            links.append(Link(a, b, "door", e, door=nd))
            nd += 1
        elif u < 0.6 and nf < sim.NF:
            frm, to = (a, b) if rng.random() < 0.5 else (b, a)
            links.append(Link(a, b, "flap", e, flap=nf, frm=frm, to=to))
            nf += 1
        else:
            links.append(Link(a, b, "open", e))
    return links


def _pits(rng, spec: Spec) -> list:
    out = []
    for r in range(spec.n_rooms):
        x0, y0, x1, y1 = spec.room_box(r)
        cen = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        n = int(rng.choice(3, p=[0.3, 0.4, 0.3]))
        for _ in range(n):
            for _ in range(30):
                w, h = rng.uniform(0.5, 1.1, 2)
                px, py = rng.uniform(x0 + 0.45, x1 - 0.45 - w), rng.uniform(y0 + 0.45, y1 - 0.45 - h)
                rc = (px, py, px + w, py + h)
                gaps = [sim.gap_centre(k) for k, _, _ in spec.room_edges(r)]
                if all(rect_dist(g[0], g[1], rc) > 1.0 for g in gaps) and rect_dist(cen[0], cen[1], rc) > 0.7 and \
                        all(rect_dist((q[0] + q[2]) / 2, (q[1] + q[3]) / 2, rc) > 0.9 for q in out):
                    out.append(rc)
                    break
    return out


def _place(rng, spec: Spec, room: int, used: list) -> tuple:
    x0, y0, x1, y1 = spec.room_box(room)
    gaps = [sim.gap_centre(k) for k, _, _ in spec.room_edges(room)]
    for _ in range(60):
        p = rng.uniform([x0 + 0.7, y0 + 0.7], [x1 - 0.7, y1 - 0.7])
        if all(np.hypot(*(p - g)) > 1.1 for g in gaps) and all(rect_dist(p[0], p[1], r) > 0.6 for r in spec.pits) \
                and all(np.hypot(*(p - q)) > 0.9 for q in used):
            used.append(p)
            return float(p[0]), float(p[1])
    raise _Reject


def connected(spec: Spec) -> bool:
    """Within every room, the gap approaches, items and spawn are reachable from each other on the planner grid
    (pits and walls only; the oracle also routes around other items)."""
    for r in range(spec.n_rooms):
        pts = [approach(spec, lk, r) for lk in spec.links if r in (lk.a, lk.b)]
        pts += [np.array(it[:2]) for it in spec.items if it[3] == r]
        if r == spec.spawn_room:
            pts.append(np.array(spec.spawn))
        if len(pts) < 2:
            continue
        box = spec.room_box(r)
        free, xs, ys = free_grid(spec, box)
        for q in pts[1:]:
            if bfs_path(free, xs, ys, pts[0], q) is None:
                return False
    return True


def generate(seed: int, gh: int, gw: int, max_tries: int = 400) -> Spec:
    path = os.path.join(CACHE, f"{seed}_{gh}x{gw}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    spec = _generate(seed, gh, gw, max_tries)
    os.makedirs(CACHE, exist_ok=True)
    tmp = f"{path}.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(spec, f)
    os.replace(tmp, path)
    return spec


def _generate(seed: int, gh: int, gw: int, max_tries: int) -> Spec:
    from maze import abstract as A
    rng = np.random.default_rng(seed)
    n = gh * gw
    for _ in range(max_tries):
        try:
            spec = Spec(gh, gw, [], [], [], (0.0, 0.0), 0, 1, seed)
            spec.links = _links(rng, gh, gw, spec)
            if not any(lk.kind != "open" for lk in spec.links):
                raise _Reject
            spec.pits = _pits(rng, spec)
            spec.spawn_room = int(rng.integers(n))
            x0, y0, x1, y1 = spec.room_box(spec.spawn_room)
            spec.spawn = ((x0 + x1) / 2, (y0 + y1) / 2)
            used = [np.array(spec.spawn)]
            items = []
            for k, r in enumerate(rng.choice([q for q in range(n) if q != spec.spawn_room], 3, replace=False)):
                items.append(_place(rng, spec, int(r), used) + (k, int(r)))
            for _ in range(int(rng.integers(0, 3))):
                r = int(rng.integers(n))
                items.append(_place(rng, spec, r, used) + (G.KEY, r))
            spec.items = items
            info = A.level_info(spec)
            if not (info["face0"] >= 2 and info["commitment"] and info["need"].count(min(info["need"])) == 1):
                raise _Reject
            order = np.argsort(np.array(info["need"]) + rng.random(3) * 0.1)  # goal order[lab] becomes label lab
            spec.items = [items[int(order[lab])][:2] + (lab, items[int(order[lab])][3]) for lab in range(3)] + \
                items[3:]
            if not connected(spec):
                raise _Reject
            return spec
        except _Reject:
            continue
    raise RuntimeError(f"no level after {max_tries} tries (seed {seed})")


def make_split(split: str, n: int, seed: int | None = None, indices=None) -> list:
    """``n`` (level, ranking) pairs for a split (only ``indices`` of them, if given; rankings stay the same)."""
    cfg = SPLITS[split]
    base = cfg["seed"] * 100_000 if seed is None else seed
    ranks = TRAIN_RANKS if cfg["ranks"] == "train" else PREF_RANKS
    rng = np.random.default_rng(base)
    picks = [ranks[rng.integers(len(ranks))] for _ in range(n)]
    idx = range(n) if indices is None else indices
    return [(generate(base + i, *cfg["g"]), picks[i]) for i in idx]


# ------------------------------------------------------------------ JAX level

def raster(spec: Spec) -> np.ndarray:
    """(5, GP, GP) int8 at GRES: wall (static, wall-edge fillers and the frame outside), door id + 1, flap id + 1,
    item id + 1, pit. Row = y, column = x, with GPAD cells of padding."""
    c = (np.arange(GP) - GPAD + 0.5) * GRES
    px, py = np.meshgrid(c, c)
    h = GRES / 2
    out = np.zeros((5, GP, GP), np.int8)
    L = sim.GM * sim.R
    wall = (px < 0) | (py < 0) | (px > L) | (py > L)
    kind, idx = edge_kinds(spec)
    rects = list(static_rects()) + [filler_rect(k) for k in range(sim.NEDGE) if kind[k] == 0]
    for r in rects:
        wall |= rect_dist(px, py, r) <= h
    out[0] = wall
    for k in range(sim.NEDGE):
        if kind[k] in (2, 3):
            m = rect_dist(px, py, filler_rect(k)) <= h
            out[1 if kind[k] == 2 else 2][m] = idx[k] + 1
    for i, (x, y, _, _) in enumerate(spec.items):
        out[3][np.hypot(px - x, py - y) <= 0.3] = i + 1
    for r in spec.pits:
        out[4][rect_dist(px, py, r) <= 0] = 1
    return out


def to_level(spec: Spec) -> G.Level:
    import jax.numpy as jnp
    gp, bp, bq = sim.base_arrays()
    _, _, _, ids = sim.model()
    kind, idx = edge_kinds(spec)
    door_xy = np.zeros((G.ND, 2), np.float32)
    door_ok = np.zeros(G.ND, bool)
    ff, ft = np.full(sim.NF, -1, np.int32), np.full(sim.NF, -1, np.int32)
    fok = np.zeros(sim.NF, bool)
    fxy, fn = np.zeros((sim.NF, 2), np.float32), np.zeros((sim.NF, 2), np.float32)
    for lk in spec.links:
        if lk.kind == "door":
            door_xy[lk.door] = sim.gap_centre(lk.edge)
            door_ok[lk.door] = True
        elif lk.kind == "flap":
            to_b = spec.frame_room(lk.to) == sim.EDGES[lk.edge][1]
            pos, quat = sim.flap_pose(lk.edge, to_b)
            bp[ids["flap_body"][lk.flap]] = pos
            bq[ids["flap_body"][lk.flap]] = quat
            ff[lk.flap], ft[lk.flap], fok[lk.flap] = lk.frm, lk.to, True
            fxy[lk.flap] = sim.gap_centre(lk.edge)
            fn[lk.flap] = sim.normal(lk.edge, to_b)
    ixy = np.zeros((G.NI, 2), np.float32)
    ik, ir = np.full(G.NI, -1, np.int32), np.full(G.NI, -1, np.int32)
    for i, (x, y, k, r) in enumerate(spec.items):
        ixy[i], ik[i], ir[i] = (x, y), k, r
    pit = np.tile(np.array([[1e3, 1e3, 1e3 + 1, 1e3 + 1]], np.float32), (G.NP, 1))
    for i, r in enumerate(spec.pits[:G.NP]):
        pit[i] = r
    f = lambda x, d=jnp.float32: jnp.asarray(np.asarray(x), d)  # noqa: E731
    return G.Level(jnp.int32(spec.gh), jnp.int32(spec.gw), f(gp), f(bp), f(bq), f(kind, jnp.int32), f(idx, jnp.int32),
                   f(door_xy), f(door_ok, bool), f(ff, jnp.int32), f(ft, jnp.int32), f(fok, bool), f(fxy), f(fn),
                   f(ixy), f(ik, jnp.int32), f(ir, jnp.int32), f(pit), f(np.array(spec.spawn)), jnp.int32(spec.keys0),
                   f(raster(spec), jnp.int8))


__all__ = ["Spec", "Link", "generate", "make_split", "to_level", "payoff", "approach", "free_grid", "bfs_path",
           "room_region", "region_mask", "TRAIN_RANKS", "PREF_RANKS", "SPLITS", "raster", "GP", "GPAD", "GRES"]
