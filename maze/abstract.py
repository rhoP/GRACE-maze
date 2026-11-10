"""The abstract layer: exact model, faces, Φmax, options, managers and the identity (adapted from GRACE-bomber).

Abstract state ``(room, keys, doors, flaps, gone, end)``: the agent's room, the keys held, the doors opened, the
flaps latched, the items taken, and the goal taken (-1 while playing). Options from a state:
- ``("go", link, dest)``: cross an open link, an opened door, or an unlatched flap from its ``frm`` room (the flap
  latches: a commitment);
- ``("open", link, room)``: open a closed adjacent door with a key (the key is spent: a commitment);
- ``("get", item, room)``: take an item in the room (a key adds a key; a goal ends the episode).

Every abstract transition the game can make is one of these options (a key can only be used at an adjacent door, and
pickups happen in the item's room), so the abstract model is exact up to simultaneous events.

Faces (the goals still reachable) by a fixpoint over the reachable abstract graph. Φmax(s) = the best payoff on the
face; a mistake is a transition that lowers it. Identity: V0 − payoff = Σ Φmax drops + Φmax at a timeout.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from maze import game as G
from maze import sim
from maze.levels import Spec

GO, OPEN, GET = "go", "open", "get"
KINDS = [GO, OPEN, GET]
END = 5
ROOMS_MAX = 16


def start(spec: Spec):
    return (spec.spawn_room, spec.keys0, 0, 0, 0, -1)


def options(spec: Spec, s) -> list[tuple]:
    room, keys, doors, flaps, gone, end = s
    if end >= 0:
        return []
    out = []
    for li, lk in enumerate(spec.links):
        if room not in (lk.a, lk.b):
            continue
        if lk.kind == "open" or (lk.kind == "door" and (doors >> lk.door) & 1):
            out.append((GO, li, lk.other(room)))
        elif lk.kind == "door":
            if keys > 0:
                out.append((OPEN, li, room))
        elif room == lk.frm and not (flaps >> lk.flap) & 1:
            out.append((GO, li, lk.to))
    for ii, it in enumerate(spec.items):
        if it[3] == room and not (gone >> ii) & 1:
            out.append((GET, ii, room))
    return out


def succ(spec: Spec, s, o):
    room, keys, doors, flaps, gone, end = s
    kind, arg, dest = o
    if kind == GO:
        lk = spec.links[arg]
        if lk.kind == "flap":
            flaps |= 1 << lk.flap
        return (dest, keys, doors, flaps, gone, end)
    if kind == OPEN:
        return (room, keys - 1, doors | (1 << spec.links[arg].door), flaps, gone, end)
    k = spec.items[arg][2]
    gone |= 1 << arg
    if k == G.KEY:
        keys = min(keys + 1, G.KMAX)
    else:
        end = k
    return (room, keys, doors, flaps, gone, end)


class Graph:
    """The reachable abstract graph of a level: faces, option distances to each exit."""

    def __init__(self, spec: Spec, s0=None):
        self.spec = spec
        s0 = start(spec) if s0 is None else s0
        self.succs, order, q = {}, [], deque([s0])
        seen = {s0}
        while q:
            s = q.popleft()
            order.append(s)
            nxt = [(o, succ(spec, s, o)) for o in options(spec, s)]
            self.succs[s] = nxt
            for _, t in nxt:
                if t not in seen:
                    seen.add(t)
                    q.append(t)
        face = {s: (frozenset([s[END]]) if s[END] >= 0 else frozenset()) for s in order}
        changed = True
        while changed:
            changed = False
            for s in reversed(order):
                if s[END] >= 0:
                    continue
                f = frozenset().union(*[face[t] for _, t in self.succs[s]]) if self.succs[s] else frozenset()
                if f != face[s]:
                    face[s], changed = f, True
        self.face = face
        pred = {s: [] for s in order}
        for s in order:
            for _, t in self.succs[s]:
                pred[t].append(s)
        self.dist = []
        for k in range(3):
            d = {s: 0 for s in order if s[END] == k}
            q = deque(d)
            while q:
                t = q.popleft()
                for s in pred[t]:
                    if s not in d:
                        d[s] = d[t] + 1
                        q.append(s)
            self.dist.append(d)

    def ensure(self, s):
        """Faces for a state outside the explored graph (a start other than the spawn)."""
        if s is not None and s not in self.face:
            g = Graph(self.spec, s)
            self.face.update(g.face)
            self.succs.update(g.succs)
            for k in range(3):
                self.dist[k].update(g.dist[k])

    def phi(self, s, p) -> float:
        if s is None:  # dead
            return 0.0
        self.ensure(s)
        if s[END] >= 0:
            return float(p[s[END]])
        return max((float(p[k]) for k in self.face[s]), default=0.0)


def level_info(spec: Spec) -> dict:
    g = Graph(spec)
    s0 = start(spec)
    commitment = any(g.face[t] < g.face[s] for s in g.succs for _, t in g.succs[s] if t[END] < 0)
    need = []  # key uses and key pickups needed before each goal: 0-1 BFS
    for k in range(3):
        best, dq = {s0: 0}, deque([s0])
        res = np.inf
        while dq:
            s = dq.popleft()
            if s[END] == k:
                res = min(res, best[s])
                continue
            for o, t in g.succs[s]:
                if t[END] >= 0 and t[END] != k:
                    continue
                w = 0 if (o[0] == GO or t[END] == k) else 1
                if best.get(t, np.inf) > best[s] + w:
                    best[t] = best[s] + w
                    (dq.appendleft if w == 0 else dq.append)(t)
        need.append(float(res))
    return {"states": len(g.face), "face0": len(g.face[s0]), "commitment": commitment, "need": need}


def table_index(room, keys, doors, flaps, gone):
    return (((room * (G.KMAX + 1) + keys) * (1 << G.ND) + doors) * (1 << sim.NF) + flaps) * (1 << G.NI) + gone


TABLE_SIZE = ROOMS_MAX * (G.KMAX + 1) * (1 << G.ND) * (1 << sim.NF) * (1 << G.NI)


def face_table(spec: Spec) -> np.ndarray:
    """uint8 bitmask of reachable goals per abstract state (the in-game check's Φmax). Exact on the reachable graph;
    other entries are 0, which is conservative. Python-side accounting (``Trace``) uses the exact ``Graph``."""
    tab = np.zeros(TABLE_SIZE, np.uint8)
    g = Graph(spec)
    for s, f in g.face.items():
        if s[END] < 0:
            tab[table_index(*s[:5])] = sum(1 << k for k in f)
    return tab


# ------------------------------------------------------------------ managers

def allowed(g: Graph, s, opts, p) -> list[int]:
    """The option shield: indices of options that keep Φmax."""
    v = g.phi(s, p)
    keep = [i for i, o in enumerate(opts) if g.phi(succ(g.spec, s, o), p) >= v - 1e-9]
    return keep or list(range(len(opts)))


def exact_choice(g: Graph, p):
    """C: keep Φmax, then the fewest options to the best exit on the face."""
    def choose(s, opts, ks):
        v = g.phi(s, p)
        best = [k for k in g.face[s] if abs(float(p[k]) - v) < 1e-9]

        def cost(i):
            t = succ(g.spec, s, opts[i])
            g.ensure(t)
            return min((g.dist[k].get(t, 10 ** 6) for k in best), default=10 ** 6)
        return min(ks, key=lambda i: (cost(i), i))
    return choose


def uniform_choice(rng):
    return lambda s, opts, ks: ks[rng.integers(len(ks))]


def no_revisit(spec: Spec, choose):
    """Wrap a choice for one episode: options whose successor is an abstract state already visited are excluded while
    another allowed option exists. A greedy manager is a function of the abstract state, so any revisit repeats
    forever (GRACE-platformer pilot). It only narrows ``ks``, so the shield and the identity
    are unchanged. Make a fresh wrapper per episode."""
    seen = set()

    def wrapped(s, opts, ks):
        seen.add(s)
        fresh = [k for k in ks if succ(spec, s, opts[k]) not in seen]
        return choose(s, opts, fresh or ks)
    return wrapped


def least_visited(spec: Spec, choose):
    """The count-based mask (added after the pilot's ``mgr`` run): keep the options whose successor has been visited
    least often this episode, and let ``choose`` pick among them. ``no_revisit`` falls back to every option once all
    successors have been seen, and a greedy manager then loops until the timeout; this was all of E+nr's remaining
    regret. It only narrows ``ks``, so the shield and the identity are unchanged. Make a fresh wrapper per episode."""
    count = {}

    def wrapped(s, opts, ks):
        count[s] = count.get(s, 0) + 1
        c = [count.get(succ(spec, s, opts[k]), 0) for k in ks]
        return choose(s, opts, [k for k, x in zip(ks, c) if x == min(c)])
    return wrapped


MASKS = {"": lambda spec, ch: ch, "nr": no_revisit, "lv": least_visited}


class Trace:
    """Φmax drops along an episode, for the identity and the mistake count."""

    def __init__(self, g: Graph, s, p):
        self.g, self.p, self.v0, self.s = g, p, g.phi(s, p), s
        self.drops, self.n_mistakes = 0.0, 0

    def after(self, t):
        d = self.g.phi(self.s, self.p) - self.g.phi(t, self.p)
        if d > 1e-9:
            self.drops += d
            self.n_mistakes += 1
        self.s = t

    def result(self, timeout: bool) -> dict:
        s = self.s
        pay = float(self.p[s[END]]) if (s is not None and s[END] >= 0) else 0.0
        t_loss = self.g.phi(s, self.p) if (timeout and s is not None and s[END] < 0) else 0.0
        regret = self.v0 - pay
        return {"v0": self.v0, "payoff": pay, "regret": regret, "drop_loss": self.drops, "timeout_loss": t_loss,
                "mistakes": self.n_mistakes, "mistake": float(self.n_mistakes > 0),
                "identity": float(abs(regret - self.drops - t_loss) < 1e-6),
                "success": float(s is not None and s[END] >= 0), "timeout": float(timeout), "dead": float(s is None)}


def run_abstract(spec: Spec, p, choose, shield: bool, max_options: int = 60, g: Graph | None = None) -> dict:
    """An episode on the abstract model (exact execution)."""
    g = g or Graph(spec)
    s = start(spec)
    tr = Trace(g, s, p)
    n = 0
    while s[END] < 0 and n < max_options:
        opts = options(spec, s)
        if not opts:
            break
        ks = allowed(g, s, opts, p) if shield else list(range(len(opts)))
        s = succ(spec, s, opts[choose(s, opts, ks)])
        tr.after(s)
        n += 1
    r = tr.result(timeout=s[END] < 0)
    r["options"] = n
    return r


__all__ = ["Graph", "start", "options", "succ", "allowed", "exact_choice", "uniform_choice", "no_revisit", "least_visited", "MASKS", "Trace",
           "run_abstract", "level_info", "face_table", "table_index", "TABLE_SIZE", "KINDS", "GO", "OPEN", "GET", "END"]
