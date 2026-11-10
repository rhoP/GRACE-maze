"""Gates before any training (``docs/plan.md`` §3).

P1 (statistical: labels = game). Per level:
1. options: from up to 12 reachable abstract states, the agent at rest at a random free point of the room, every
   option under the oracle reaches exactly ``abstract.succ`` (else a plan failure, a stall, or a wrong outcome);
2. flaps are one-way: from the *to* side (unlatched, and after a crossing has latched it) rollouts of random
   actions, a PD controller aimed through the gap and a full-force ram never reach the *from* room;
3. doors need a key: without one, 30-tick rollouts of the same controllers with ``use`` pressed never cross or open
   it; with one, a single use opens the door and spends exactly one key.
Rollouts start with every item gone, so no key can be picked up on the way.
P2 (abstract model): C has 0 regret and 0 mistakes, D 0 mistakes, the identity on every episode.

    PYTHONPATH=. python scripts/gates.py --p1 --p2 --p1-levels 5 --out runs/gates
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from maze import abstract as A
from maze import game as G
from maze import levels as L
from maze import oracle as O
from maze import sim

ROLL = 60  # ticks per flap rollout
DOOR_ROLL = 30  # door rollouts: too short to reach the other room around through two other gaps
ALL_GONE = (1 << G.NI) - 1  # no pickups during the rollouts (a picked-up key would open the door legitimately)
N_ROLL = 64  # rollouts per controller and condition


def rest_state(spec, lv, s, rng):
    room, keys, doors, flaps, gone, _ = s
    free, xs, ys = L.free_grid(spec, spec.room_box(room), doors, flaps, [i for i in range(len(spec.items))])
    fy, fx = np.nonzero(free)
    k = rng.integers(len(fy))
    q = np.zeros(2 + sim.NF, np.float32)
    q[:2] = xs[fx[k]], ys[fy[k]]
    i32 = jnp.int32
    return G.State(jnp.asarray(q), jnp.zeros(2 + sim.NF), i32(0), i32(room), i32(keys), i32(doors), i32(flaps),
                   i32(gone), i32(-1), jnp.asarray(True))


@partial(jax.jit, static_argnames="ticks")
def _rollouts(lv, st0, ctrl, aim, start_xy, keys, ticks=ROLL):
    """Per rollout: whether the room ever equals aim[2] during ROLL ticks, and the final doors and keys.
    ctrl 0 random, 1 PD toward aim[:2], 2 ram along aim[3:5] (with a random tangent); ``use`` always pressed."""
    def one(start_xy, k):
        st = st0._replace(qpos=st0.qpos.at[:2].set(start_xy))

        def body(c, kk):
            st, hit = c
            pos, vel = st.qpos[:2], st.qvel[:2]
            rnd = jax.random.uniform(kk, (2,), minval=-1, maxval=1)
            pd = jnp.clip(2.0 * (aim[:2] - pos) - 0.5 * vel, -1, 1)
            ram = jnp.clip(aim[3:5] + 0.5 * rnd * aim[3:5][::-1], -1, 1)
            u = jnp.where(ctrl == 0, rnd, jnp.where(ctrl == 1, pd, ram))
            st = G.step(lv, st, jnp.concatenate([u, jnp.ones(1)]))
            return (st, hit | (st.room == aim[2].astype(jnp.int32))), None
        (st, hit), _ = jax.lax.scan(body, (st, jnp.asarray(False)), jax.random.split(k, ticks))
        return hit, st.doors, st.keys
    return jax.vmap(one)(start_xy, keys)


def adversarial(lv, st0, start_xy, ctrl, aim, seed, ticks=ROLL):
    xy = jnp.asarray(np.asarray(start_xy), jnp.float32)
    return _rollouts(lv, st0, jnp.int32(ctrl), jnp.asarray(aim, jnp.float32), xy,
                     jax.random.split(jax.random.PRNGKey(seed), xy.shape[0]), ticks=ticks)


def starts_near(c, n_out, rng, n=N_ROLL):
    """Random starts on the n_out side of a gap centred at c, 0.4 to 1.4 from the wall, within ±0.9 along it."""
    t = n_out[::-1]
    d = rng.uniform(0.4, 1.4, n)
    s = rng.uniform(-0.9, 0.9, n)
    return c[None] + n_out[None] * d[:, None] + t[None] * s[:, None]


def p1_level(spec, rng) -> dict:
    lv = L.to_level(spec)
    g = A.Graph(spec)
    out = {"options": 0, "plan_fail": 0, "stall": 0, "wrong": 0, "flap_rollouts": 0, "flap_leaks": 0,
           "door_rollouts": 0, "door_leaks": 0, "door_use_errors": 0}
    states = [s for s in g.face if s[A.END] < 0]
    for si in rng.permutation(len(states))[:12]:
        s = states[si]
        for o in A.options(spec, s):
            st = rest_state(spec, lv, s, rng)
            r = O.run_option(spec, lv, st, s, o)
            out["options"] += 1
            if r is None:
                out["plan_fail"] += 1
                continue
            fin, used, ok, _ = r
            if not ok:
                got = O.abstract_of(fin)
                out["stall" if got == s else "wrong"] += 1
    base = G.initial(lv)._replace(gone=jnp.int32(ALL_GONE))
    for lk in spec.flaps:
        c = sim.gap_centre(lk.edge)
        n_to = sim.normal(lk.edge, spec.frame_room(lk.to) == sim.EDGES[lk.edge][1])
        for latched in (False, True):
            st0 = base._replace(room=jnp.int32(lk.to), flaps=jnp.int32((1 << lk.flap) if latched else 0))
            xy = starts_near(c, n_to, rng)
            tgt = c - n_to * 1.0
            for ctrl in (0, 1, 2):
                hit, _, _ = adversarial(lv, st0, xy, ctrl, [tgt[0], tgt[1], lk.frm, -n_to[0], -n_to[1]],
                                        int(rng.integers(1 << 30)))
                out["flap_rollouts"] += len(xy)
                out["flap_leaks"] += int(np.asarray(hit).sum())
    for lk in spec.doors:
        c = sim.gap_centre(lk.edge)
        for room in (lk.a, lk.b):
            n_out = sim.normal(lk.edge, spec.frame_room(room) == sim.EDGES[lk.edge][1])
            other = lk.other(room)
            xy = starts_near(c, n_out, rng)
            tgt = c - n_out * 1.0
            st0 = base._replace(room=jnp.int32(room), keys=jnp.int32(0))
            for ctrl in (0, 1, 2):
                hit, doors, _ = adversarial(lv, st0, xy, ctrl, [tgt[0], tgt[1], other, -n_out[0], -n_out[1]],
                                            int(rng.integers(1 << 30)), ticks=DOOR_ROLL)
                out["door_rollouts"] += len(xy)
                out["door_leaks"] += int(np.asarray(hit).sum() + (np.asarray(doors) != 0).sum())
            st1 = base._replace(room=jnp.int32(room), keys=jnp.int32(2))
            _, doors, keys = adversarial(lv, st1, c[None] + n_out[None] * 0.4, 1, [tgt[0], tgt[1], other,
                                                                                    -n_out[0], -n_out[1]], 0)
            ok = int(np.asarray(doors)[0]) == (1 << lk.door) and int(np.asarray(keys)[0]) == 1
            out["door_use_errors"] += int(not ok)
    return out


def p2(n: int) -> dict:
    res = {}
    for sp in L.SPLITS:
        rows = {"C": [], "D": [], "U": []}
        for i, (spec, rank) in enumerate(L.make_split(sp, n)):
            p = L.payoff(rank)
            g = A.Graph(spec)
            rows["C"].append(A.run_abstract(spec, p, A.exact_choice(g, p), True, 60, g))
            rows["D"].append(A.run_abstract(spec, p, A.uniform_choice(np.random.default_rng(i)), True, 60, g))
            rows["U"].append(A.run_abstract(spec, p, A.uniform_choice(np.random.default_rng(i)), False, 60, g))
        res[sp] = {arm: {k: float(np.mean([r[k] for r in rs])) for k in ("regret", "mistake", "timeout", "identity")}
                   | {"options_p95": float(np.percentile([r["options"] for r in rs], 95))}
                   for arm, rs in rows.items()}
        res[sp]["pass"] = (res[sp]["C"]["regret"] == 0 and res[sp]["C"]["mistake"] == 0
                           and res[sp]["D"]["mistake"] == 0
                           and min(res[sp][a]["identity"] for a in rows) == 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", action="store_true")
    ap.add_argument("--p2", action="store_true")
    ap.add_argument("--p1-levels", type=int, default=5)
    ap.add_argument("--p2-levels", type=int, default=50)
    ap.add_argument("--task", type=int, default=None, help="with --split-task: one split per array task")
    ap.add_argument("--split-task", action="store_true")
    ap.add_argument("--out", default="runs/gates")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.p1:
        splits = [list(L.SPLITS)[a.task]] if a.split_task else list(L.SPLITS)
        for sp in splits:
            t0 = time.time()
            rng = np.random.default_rng(L.SPLITS[sp]["seed"])
            tot = {}
            for spec, _ in L.make_split(sp, a.p1_levels):
                for k, v in p1_level(spec, rng).items():
                    tot[k] = tot.get(k, 0) + v
            tot["time"] = round(time.time() - t0)
            tot["leaks_pass"] = tot["flap_leaks"] == 0 and tot["door_leaks"] == 0 and tot["door_use_errors"] == 0
            print(sp, json.dumps(tot), flush=True)
            (out / f"p1_{sp}.json").write_text(json.dumps(tot, indent=1))
    if a.p2:
        r = p2(a.p2_levels)
        print(json.dumps({sp: r[sp]["pass"] for sp in r}), flush=True)
        (out / "p2.json").write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
