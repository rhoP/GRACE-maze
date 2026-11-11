"""Tick-by-tick trace of G+lv on one level (pilot models): per option the start/end abstract state, and per tick
the candidate tests (alive, keeps Φmax, slack of the state and of each successor) when the check intervenes."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from maze import abstract as A
from maze import envs as E
from maze import game as G
from maze import hier as Hh
from maze import levels as L
from maze import manager as M
from maze import oracle as O
from maze.ppo import ActorCritic

run, split, idx = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
maxprint = int(sys.argv[4]) if len(sys.argv) > 4 else 12
import os
ALL = bool(os.environ.get("ALL"))
args = json.loads((run / "result.json").read_text())["args"]
with open(run / "skill.pkl", "rb") as f:
    params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
mgr = M.Manager(args["seed"])
with open(run / "manager.pkl", "rb") as f:
    mgr.params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
spec, rank = L.make_split(split, idx + 1)[idx]
p = L.payoff(rank)
g = A.Graph(spec)
lv = L.to_level(spec)
table = jnp.asarray(A.face_table(spec))
pj = jnp.asarray(p, jnp.float32)
net = ActorCritic(n_actions=G.N_ACT, width=args["width"])


@jax.jit
def tick(st, target, otype):
    a = net.apply(params, E.skill_obs(lv, st, target, otype))[0][0]
    v0 = Hh.phi_j(table, pj, st)
    z = Hh.zones(lv, table, pj, st)
    waive = Hh.slack(st, z)
    nxt = jax.vmap(lambda u: G.step(lv, st, u))(Hh.candidates(st, a, z))
    alive = nxt.alive
    keep = jax.vmap(lambda s1: Hh.phi_j(table, pj, s1) >= v0 - 1e-6)(nxt)
    sf = jax.vmap(lambda s1: Hh.slack(s1, z))(nxt)
    st2, br = Hh.checked_step(lv, table, pj, st, a)
    (rect, r_act), (cen, c_act) = z
    return st2, br, a, alive, keep, sf, waive, v0, r_act, c_act


print("payoff", p, "spawn room", spec.spawn_room, "links", [(l.a, l.b, l.kind) for l in spec.links])
print("items", spec.items)
choose = A.MASKS["lv"](spec, mgr.choice(spec, p))
st = G.initial(lv)
s = A.start(spec)
printed, t = 0, 0
while s is not None and s[A.END] < 0 and t < 1500:
    opts = A.options(spec, s)
    keep_i = A.allowed(g, s, opts, p)
    opts_k = [opts[k] for k in keep_i]
    o = opts_k[choose(s, opts_k, list(range(len(opts_k))))]
    tgt = A.succ(spec, s, o)
    print(f"t={t} s={s} phi={g.phi(s, p):.2f} option={o} -> {tgt} phi={g.phi(tgt, p):.2f}")
    target = jnp.asarray(O.waypoint(spec, s, o), jnp.float32)
    for k in range(Hh.SKILL_LIMIT):
        st, br, a, alive, keep, sf, waive, v0, r_act, c_act = tick(st, target, jnp.int32(O.option_type(o)))
        t += 1
        s2 = O.abstract_of(st) if hasattr(O, "abstract_of") else None
        if (bool(br) or ALL) and printed < maxprint:
            printed += 1
            print(f"  t={t} pos={np.round(np.asarray(st.qpos[:2]), 2)} vel={np.round(np.asarray(st.qvel[:2]), 2)} "
                  f"a={np.round(np.asarray(a), 2)} slack={float(waive):.3f} flap={np.round(np.asarray(st.qpos[2:3]), 2)} alive={np.asarray(alive).astype(int)} "
                  f"keep={np.asarray(keep).astype(int)} slack1={np.round(np.asarray(sf), 3)} "
                  f"zones r={np.flatnonzero(np.asarray(r_act))} items={np.flatnonzero(np.asarray(c_act))}")
        ok, fail = E.judge(st, jnp.asarray(s, jnp.int32), jnp.asarray(tgt, jnp.int32))
        if bool(ok) or bool(fail):
            break
    s_new = tuple(int(v) for v in np.asarray(G.abstract_j(st))) if bool(st.alive) else None
    print(f"  end t={t} state={s_new} phi={g.phi(s_new, p) if s_new else 0:.2f} ok={bool(ok)}")
    s = s_new
