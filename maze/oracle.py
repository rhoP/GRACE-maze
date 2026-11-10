"""Options in the game: waypoints, the oracle executor (grid planner + PD path follower), and episodes.

An option succeeds when the first abstract change is the expected one (``abstract.succ``) with the agent alive. The
oracle plans a path on the planner grid (``levels.free_grid``) inside the current room, extended through the
option's gap, around pits and every other present item. A PD controller follows it with a speed cap (1.2, top speed
2), and presses ``use`` within 0.6 of the door for an ``open`` option.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from maze import abstract as A
from maze import game as G
from maze import levels as L
from maze import sim
from maze.levels import Spec

OPT_TYPES = ["go", "open", "get"]
MAXT = 240  # ticks per oracle option
P = 96  # path points


def option_type(o) -> int:
    return A.KINDS.index(o[0])


def abstract_of(st: G.State):
    if not bool(st.alive):
        return None
    return tuple(int(v) for v in np.asarray(G.abstract_j(st)))


def waypoint(spec: Spec, s, o) -> tuple[float, float]:
    """The point the option heads for: 0.6 past the gap into the destination, 0.45 before a door, or the item."""
    kind, arg, dest = o
    if kind == A.GET:
        return tuple(float(v) for v in spec.items[arg][:2])
    lk = spec.links[arg]
    if kind == A.OPEN:
        return tuple(float(v) for v in L.approach(spec, lk, s[0], 0.45))
    return tuple(float(v) for v in L.approach(spec, lk, dest, 0.6))


def plan_path(spec: Spec, st: G.State, s, o):
    """(P, 2) path points and the count, or None if the planner finds no path."""
    kind, arg, dest = o
    room, keys, doors, flaps, gone, end = s
    lk = spec.links[arg] if kind in (A.GO, A.OPEN) else None
    box = L.room_region(spec, room, lk if kind == A.GO else None)
    avoid = [i for i in range(len(spec.items)) if not (gone >> i) & 1 and not (kind == A.GET and i == arg)]
    free, xs, ys = L.free_grid(spec, box, doors, flaps, avoid)
    pos = np.asarray(st.qpos[:2], float)
    free &= L.region_mask(spec, room, lk if kind == A.GO else None, xs, ys, pos)
    tgt = np.array(waypoint(spec, s, o))
    path = L.bfs_path(free, xs, ys, pos, tgt, start_tol=0.8)
    if path is None:
        return None
    pts = np.array(path[::3] + [tuple(tgt)], np.float32)
    if len(pts) > P:
        pts = pts[np.linspace(0, len(pts) - 1, P).astype(int)]
    out = np.repeat(pts[-1:], P, 0)
    out[:len(pts)] = pts
    return out, len(pts)


def pd_action(st: G.State, path, n, k, use_xy, use_on):
    pos, vel = st.qpos[:2], st.qvel[:2]
    k = jnp.where((jnp.linalg.norm(path[k] - pos) < 0.3) & (k < n - 1), k + 1, k)
    carrot = path[jnp.minimum(k + 1, n - 1)]
    to = carrot - pos
    d_final = jnp.linalg.norm(path[n - 1] - pos)
    vcap = jnp.clip(0.3 + d_final, 0.3, 1.2)
    v_des = vcap * to / (jnp.linalg.norm(to) + 1e-6)
    u = jnp.clip(sim.DAMP * v_des / sim.GEAR + 1.5 * (v_des - vel), -1.0, 1.0)
    use = use_on & (jnp.linalg.norm(pos - use_xy) < 0.6)
    return jnp.concatenate([u, jnp.where(use, 1.0, -1.0)[None]]), k


def judge(st: G.State, s_abs, t_abs):
    """(ok, fail) after a step: the expected abstract change with the agent alive, or anything else."""
    ab = G.abstract_j(st)
    at_s = (ab == s_abs).all()
    at_t = (ab == t_abs).all()
    return at_t & st.alive, ~st.alive | (~at_s & ~at_t)


@lru_cache(maxsize=None)
def _runner():
    def run(lv, st, path, n, use_xy, use_on, s_abs, t_abs):
        def body(c, _):
            st, k, done, ok, t = c
            a, k2 = pd_action(st, path, n, k, use_xy, use_on)
            st2 = G.step(lv, st, a)
            ok2, fail = judge(st2, s_abs, t_abs)
            st2 = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), st, st2)
            return (st2, jnp.where(done, k, k2), done | ok2 | fail, ok | (ok2 & ~done),
                    t + (~done).astype(jnp.int32)), st2
        (st, _, _, ok, t), traj = jax.lax.scan(
            body, (st, jnp.int32(0), jnp.asarray(False), jnp.asarray(False), jnp.int32(0)), None, MAXT)
        return st, t, ok, traj
    return jax.jit(run)


FAILS: list = []


def run_option(spec: Spec, lv: G.Level, st: G.State, s, o):
    """The oracle on option ``o``: (final state, ticks, ok, trajectory of states (MAXT,)) or None if unplanned."""
    pl = plan_path(spec, st, s, o)
    if pl is None:
        FAILS.append((spec.seed, s, o, "plan"))
        return None
    path, n = pl
    t = A.succ(spec, s, o)
    use_on = o[0] == A.OPEN
    use_xy = sim.gap_centre(spec.links[o[1]].edge) if use_on else np.zeros(2)
    fin, used, ok, traj = _runner()(lv, st, jnp.asarray(path), jnp.int32(n), jnp.asarray(use_xy, jnp.float32),
                                    jnp.asarray(use_on), jnp.asarray(s, jnp.int32), jnp.asarray(t, jnp.int32))
    if not bool(ok):
        FAILS.append((spec.seed, s, o, "exec"))
    return fin, int(used), bool(ok), traj


def oracle_executor(spec, lv, st, s, o):
    r = run_option(spec, lv, st, s, o)
    if r is None:
        return st, 0, []
    fin, used, ok, traj = r
    return fin, used, [jax.tree_util.tree_map(lambda x: x[i], traj) for i in range(used)]


def run_game(spec: Spec, p, choose, shield: bool, executor, budget: int, g: A.Graph | None = None,
             record: bool = False, max_options: int = 200, lv: G.Level | None = None) -> dict:
    """An episode through the game. ``executor(spec, lv, st, s, o) -> (final state, ticks used, states)``."""
    g = g or A.Graph(spec)
    lv = lv if lv is not None else L.to_level(spec)
    st = G.initial(lv)
    s = A.start(spec)
    tr = A.Trace(g, s, p)
    ticks, n_opt, stalls, banned = 0, 0, 0, set()
    states = [st] if record else None
    while s is not None and s[A.END] < 0 and ticks < budget and n_opt < max_options:
        opts = A.options(spec, s)
        if shield:  # keep Φmax first; a stalled option is excluded only while another Φmax-keeping one exists
            keep = [opts[k] for k in A.allowed(g, s, opts, p)]
            fresh = [o for o in keep if o not in banned]
            opts = fresh or keep
        else:
            opts = [o for o in opts if o not in banned] or opts
        if not opts:
            break
        o = opts[choose(s, opts, list(range(len(opts))))]
        st2, used, traj = executor(spec, lv, st, s, o)
        n_opt += 1
        ticks += used
        if record:
            states.extend(traj)
        s2 = abstract_of(st2)
        if s2 == s:
            stalls += 1
            banned.add(o)
        else:
            banned = set()
            tr.after(s2)
        st, s = st2, s2
    r = tr.result(timeout=s is not None and s[A.END] < 0)
    r.update(ticks=ticks, options=n_opt, stalls=stalls)
    if record:
        r["states"] = states
    return r


__all__ = ["run_option", "oracle_executor", "run_game", "abstract_of", "waypoint", "plan_path", "pd_action", "judge",
           "option_type", "OPT_TYPES", "MAXT"]
