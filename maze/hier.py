"""Arms that run in the game with learned policies: the skill executor (H, G) and the flat agent (A, A:shield).

The in-game check (G, A:shield) is a braking-distance test, the continuous analogue of the bomber's ``viable``. Each
tick the policy's action is simulated; it is kept when the successor
- is alive and does not lower Φmax (the level's face table, ``abstract.face_table``), and
- is *safe*: the box swept by the move plus the full braking distance on each axis (bang-bang braking against the
  damping, closed form) meets no forbidden zone.
Otherwise the brake action is applied. Forbidden zones are the pits, plus the trigger zone of every event that would
lower Φmax from the current abstract state: crossing an unlatched flap from this room (a rectangle past the gap's
centreline), or picking up an item (its pickup disc). A key use is an action, not a motion, so the Φmax test on the
successor covers it. From a safe state braking stays safe up to the contact and integration error, which ``tests``
bound against MuJoCo rollouts.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from maze import abstract as A
from maze import envs as E
from maze import game as G
from maze import oracle as O
from maze import sim
from maze.ppo import ActorCritic

SKILL_LIMIT = 300
MARGIN = 0.05


def phi_parts(table, p, room, keys, doors, flaps, gone):
    face = table[A.table_index(room, jnp.clip(keys, 0, G.KMAX), doors, flaps, gone)]
    return jnp.max(jnp.where(((face >> jnp.arange(3)) & 1) > 0, p, 0.0))


def phi_j(table, p, st: G.State):
    v = phi_parts(table, p, st.room, st.keys, st.doors, st.flaps, st.gone)
    v = jnp.where(st.end >= 0, p[jnp.maximum(st.end, 0)], v)
    return jnp.where(st.alive, v, 0.0)


def stop_dist(v):
    """Braking distance from speed |v| on one axis: m/c |v| − F m / c² ln(1 + c |v| / F)."""
    a = jnp.abs(v)
    m, c, f = sim.MASS, sim.DAMP, sim.GEAR
    return jnp.sign(v) * (m / c * a - f * m / c ** 2 * jnp.log1p(c * a / f))


def zones(lv: G.Level, table, p, st: G.State):
    """(Z, 4) rectangles and (Z,) active flags: pits, Φmax-lowering flap crossings, Φmax-lowering pickups."""
    v0 = phi_j(table, p, st)
    latched = G.bits(st.flaps, sim.NF)
    f_act = lv.flap_ok & ~latched & (lv.flap_from == st.room)
    f_phi = jax.vmap(lambda to, f: phi_parts(table, p, to, st.keys, st.doors, st.flaps | (1 << f), st.gone))(
        jnp.maximum(lv.flap_to, 0), jnp.arange(sim.NF))
    c, n = lv.flap_xy, lv.flap_n
    tang = jnp.abs(n[:, ::-1]) * (sim.GAP / 2)  # half extent along the wall
    far = c + n * 0.4  # from the wall's centreline 0.4 into the flap's to room
    f_rect = jnp.concatenate([jnp.minimum(c, far) - tang, jnp.maximum(c, far) + tang], 1)
    present = ~G.bits(st.gone, G.NI) & (lv.item_kind >= 0)
    is_key = lv.item_kind == G.KEY
    i_phi = jax.vmap(lambda i, k, key: jnp.where(
        key, phi_parts(table, p, st.room, st.keys + 1, st.doors, st.flaps, st.gone | (1 << i)),
        p[jnp.clip(k, 0, 2)]))(jnp.arange(G.NI), lv.item_kind, is_key)
    rect = jnp.concatenate([lv.pit, f_rect], 0)
    r_act = jnp.concatenate([jnp.ones(G.NP, bool), f_act & (f_phi < v0 - 1e-6)])
    return (rect, r_act), (lv.item_xy, present & (i_phi < v0 - 1e-6))


ITEM_MARGIN = 0.02


def safe(st: G.State, z):
    """The box from ``st``'s position to its full braking stop meets no active zone: rectangles (pits, flap
    crossings; with MARGIN) and item discs (radius PICK + ITEM_MARGIN, the pickup's own trigger). A tick moves at
    most 0.2, so the swept segment between ticks is covered by the previous tick's stop box."""
    (rect, r_act), (cen, c_act) = z
    p1 = st.qpos[:2]
    p2 = p1 + stop_dist(st.qvel[:2])
    lo, hi = jnp.minimum(p1, p2), jnp.maximum(p1, p2)
    hit = (lo[0] - MARGIN < rect[:, 2]) & (hi[0] + MARGIN > rect[:, 0]) & (lo[1] - MARGIN < rect[:, 3]) & \
        (hi[1] + MARGIN > rect[:, 1])
    d = jnp.linalg.norm(jnp.clip(cen, lo, hi) - cen, axis=1)
    return ~(hit & r_act).any() & ~((d < G.PICK + ITEM_MARGIN) & c_act).any()


def brake_action(st: G.State):
    return jnp.concatenate([-jnp.clip(st.qvel[:2] / 0.1, -1.0, 1.0), jnp.array([-1.0])])


ANGLES = (0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2)


def candidates(st: G.State, a):
    """The policy's action, the same force rotated by ±45° and ±90° (``use`` kept), then the brake."""
    c, s_ = jnp.cos(jnp.asarray(ANGLES)), jnp.sin(jnp.asarray(ANGLES))
    f = jnp.stack([c * a[0] - s_ * a[1], s_ * a[0] + c * a[1]], 1)
    rot = jnp.concatenate([f, jnp.full((len(ANGLES), 1), a[2])], 1)
    return jnp.concatenate([rot, brake_action(st)[None]], 0)


def checked_step(lv: G.Level, table, p, st: G.State, a):
    """(next state, intervened): the first allowed candidate (``candidates``), else the brake.

    A candidate is allowed when its successor is alive, keeps Φmax, and is safe. If ``st`` itself already fails the
    (conservative) safety test, the safety part is waived and only the exact tests remain: braking cannot leave such
    a state (the smoke run: braking on 58% of ticks on mpref). Rotated candidates let the agent slide round a zone
    that the policy pushes into, where pure braking left it pushing into the same zone every tick."""
    v0 = phi_j(table, p, st)
    z = zones(lv, table, p, st)
    waive = ~safe(st, z)
    cand = candidates(st, a)
    nxt = jax.vmap(lambda u: G.step(lv, st, u))(cand)
    ok = jax.vmap(lambda s1: s1.alive & (phi_j(table, p, s1) >= v0 - 1e-6) & (safe(s1, z) | waive))(nxt)
    ok = ok.at[-1].set(True)
    k = jnp.argmax(ok)
    return jax.tree_util.tree_map(lambda x: x[k], nxt), k > 0


@lru_cache(maxsize=None)
def _skill_runner(width: int):
    net = ActorCritic(n_actions=G.N_ACT, width=width)

    def run(params, lv, table, p, st, s_abs, t_abs, target, otype, shield):
        def cond(c):
            return ~c[2] & (c[1] < SKILL_LIMIT)

        def body(c):
            st, t, done, ok, nb = c
            a = net.apply(params, E.skill_obs(lv, st, target, otype))[0][0]
            st_c, braked = checked_step(lv, table, p, st, a)
            st2 = jax.tree_util.tree_map(lambda x, y: jnp.where(shield, x, y), st_c, G.step(lv, st, a))
            ok2, fail = E.judge(st2, s_abs, t_abs)
            return st2, t + 1, ok2 | fail, ok2, nb + (shield & braked).astype(jnp.int32)

        st, t, done, ok, nb = jax.lax.while_loop(
            cond, body, (st, jnp.int32(0), jnp.asarray(False), jnp.asarray(False), jnp.int32(0)))
        return st, t, ok, nb

    return jax.jit(run)


def skill_executor(params, spec, p, shield: bool, width: int = 256, log: list | None = None):
    """An executor for ``oracle.run_game`` that runs the learned skill's mean action (H: shield False, G: True)."""
    table = jnp.asarray(A.face_table(spec))
    pj = jnp.asarray(p, jnp.float32)
    run = _skill_runner(width)

    def executor(spec_, lv, st, s, o):
        t = A.succ(spec_, s, o)
        st2, used, ok, nb = run(params, lv, table, pj, st, jnp.asarray(s, jnp.int32), jnp.asarray(t, jnp.int32),
                                jnp.asarray(O.waypoint(spec_, s, o), jnp.float32), jnp.int32(O.option_type(o)),
                                jnp.asarray(shield))
        if log is not None:
            log.append({"otype": O.option_type(o), "ok": bool(ok), "ticks": int(used), "dead": not bool(st2.alive),
                        "braked": int(nb)})
        return st2, int(used), []

    return executor


# ------------------------------------------------------------------ flat agent

@lru_cache(maxsize=None)
def _flat_runner(width: int, budget: int):
    net = ActorCritic(n_actions=G.N_ACT, width=width)

    def run(params, bank, table, li, p, shield):
        lv = E.level(bank, li)
        st = G.initial(lv)

        def body(st, _):
            a = net.apply(params, E.flat_obs(lv, st, p))[0][0]
            st_c, braked = checked_step(lv, table, p, st, a)
            st2 = jax.tree_util.tree_map(lambda x, y: jnp.where(shield, x, y), st_c, G.step(lv, st, a))
            ab = jnp.where(st2.alive, G.abstract_j(st2), -9)
            return st2, (ab, shield & braked)

        st, (abs_, br) = jax.lax.scan(body, st, None, budget)
        return abs_, br

    return jax.jit(jax.vmap(run, in_axes=(None, None, 0, 0, 0, None)))


def eval_flat(params, pairs, shield: bool, budget: int, width: int = 256) -> list[dict]:
    """Flat episodes (mean action); the accounting runs in Python on the per-tick abstract states with the Graph."""
    from maze.levels import payoff
    specs = [s for s, _ in pairs]
    bank = E.make_bank(specs)
    tables = jnp.asarray(np.stack([A.face_table(s) for s in specs]))
    ps = jnp.asarray(np.stack([payoff(r) for _, r in pairs]))
    abs_, br = _flat_runner(width, budget)(params, bank, tables, jnp.arange(len(specs)), ps, jnp.asarray(shield))
    abs_, br = np.asarray(abs_), np.asarray(br)
    out = []
    for i, (spec, rank) in enumerate(pairs):
        p = payoff(rank)
        g = A.Graph(spec)
        s = A.start(spec)
        tr = A.Trace(g, s, p)
        n = budget
        for t, row in enumerate(abs_[i]):
            nxt = None if row[0] == -9 else tuple(int(v) for v in row)
            if nxt != s:
                tr.after(nxt)
                s = nxt
            if s is None or s[A.END] >= 0:
                n = t + 1
                break
        r = tr.result(timeout=s is not None and s[A.END] < 0)
        r["braked"] = float(br[i, :n].mean())
        out.append(r)
    return out


__all__ = ["skill_executor", "eval_flat", "phi_j", "checked_step", "zones", "safe", "stop_dist", "brake_action"]
