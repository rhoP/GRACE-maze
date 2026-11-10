"""Figures: level maps with the room graph (one per split), and a GIF of an episode (the oracle, or G+lv with the
learned skill under the braking check when ``--run`` gives a trained seed folder).

    PYTHONPATH=. python scripts/figures.py --out plots
    PYTHONPATH=. python scripts/figures.py --out plots --run runs/pilot/seed0 --split mpref --index 0
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from maze import abstract as A  # noqa: E402
from maze import envs as E  # noqa: E402
from maze import game as G  # noqa: E402
from maze import hier as Hh  # noqa: E402
from maze import levels as L  # noqa: E402
from maze import manager as M  # noqa: E402
from maze import oracle as O  # noqa: E402
from maze import render as R  # noqa: E402
from maze.ppo import ActorCritic  # noqa: E402


def skill_episode(spec, p, params, mgr, width: int, budget: int = 1500):
    """G+lv tick by tick (recorded): the manager picks options, the skill acts under the braking check."""
    lv = L.to_level(spec)
    table = jnp.asarray(A.face_table(spec))
    pj = jnp.asarray(p, jnp.float32)
    net = ActorCritic(n_actions=G.N_ACT, width=width)
    act = jax.jit(lambda st, tgt, ot: Hh.checked_step(
        lv, table, pj, st, net.apply(params, E.skill_obs(lv, st, tgt, ot))[0][0])[0])
    g = A.Graph(spec)
    choose = A.least_visited(spec, mgr.choice(spec, p))
    st = G.initial(lv)
    s = A.start(spec)
    states = [st]
    while s is not None and s[A.END] < 0 and len(states) < budget:
        opts = A.options(spec, s)
        keep = [opts[k] for k in A.allowed(g, s, opts, p)]
        o = keep[choose(s, keep, list(range(len(keep))))]
        t = A.succ(spec, s, o)
        tgt = jnp.asarray(O.waypoint(spec, s, o), jnp.float32)
        for _ in range(Hh.SKILL_LIMIT):
            st = act(st, tgt, jnp.int32(O.option_type(o)))
            states.append(st)
            ok, fail = E.judge(st, jnp.asarray(s, jnp.int32), jnp.asarray(t, jnp.int32))
            if bool(ok) or bool(fail):
                break
        s = O.abstract_of(st)
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots")
    ap.add_argument("--run", default="")
    ap.add_argument("--split", default="mpref")
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, sp in zip(axes, L.SPLITS):
        spec, rank = L.make_split(sp, 1)[0]
        R.level_map(spec, ax, f"{sp}  (payoff {'/'.join(str(v) for v in L.payoff(rank))} for A/B/C)")
    fig.tight_layout()
    fig.savefig(out / "levels.png", dpi=150)
    spec, rank = L.make_split(a.split, a.index + 1, indices=[a.index])[0]
    p = L.payoff(rank)
    if a.run:
        run = Path(a.run)
        args = json.loads((run / "result.json").read_text())["args"]
        with open(run / "skill.pkl", "rb") as f:
            params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
        mgr = M.Manager(0)
        with open(run / "manager.pkl", "rb") as f:
            mgr.params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
        states = skill_episode(spec, p, params, mgr, args["width"])
        name = f"episode_G_{a.split}{a.index}.gif"
    else:
        g = A.Graph(spec)
        states = O.run_game(spec, p, A.exact_choice(g, p), True, O.oracle_executor, 3000, g, record=True)["states"]
        name = f"episode_oracle_{a.split}{a.index}.gif"
    R.gif(spec, states, str(out / name), every=2)
    print("wrote", out / "levels.png", out / name, len(states), "ticks")


if __name__ == "__main__":
    main()
