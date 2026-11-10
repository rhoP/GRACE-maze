"""G+lv per episode (the game as now; ``--old`` changes only the check) with the pilot's saved manager and skill: mistakes, braked share, and for each option whether
the check intervened. ``--old`` restores the pilot's check (brake whenever no candidate is safe and Φmax-keeping).

    PYTHONPATH=. python scripts/check_diag.py --run runs/pilot/seed1 --split mpref [--old] [--levels 50]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from maze import abstract as A
from maze import hier as Hh
from maze import levels as L
from maze import manager as M
from maze import oracle as O

BUDGET = {"mtest": 1500, "mpref": 1500, "mbig": 2500}


def old_checked_step(lv, table, p, st, a):
    v0 = Hh.phi_j(table, p, st)
    z = Hh.zones(lv, table, p, st)
    waive = ~Hh.safe(st, z)
    ang = jnp.asarray(Hh.ANGLES)
    f = jnp.stack([jnp.cos(ang) * a[0] - jnp.sin(ang) * a[1], jnp.sin(ang) * a[0] + jnp.cos(ang) * a[1]], 1)
    cand = jnp.concatenate([jnp.concatenate([f, jnp.full((len(Hh.ANGLES), 1), a[2])], 1),
                            Hh.brake_action(st)[None]], 0)
    nxt = jax.vmap(lambda u: Hh.G.step(lv, st, u))(cand)
    ok = jax.vmap(lambda s1: s1.alive & (Hh.phi_j(table, p, s1) >= v0 - 1e-6) & (Hh.safe(s1, z) | waive))(nxt)
    k = jnp.argmax(ok.at[-1].set(True))
    return jax.tree_util.tree_map(lambda x: x[k], nxt), k > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="mpref")
    ap.add_argument("--levels", type=int, default=50)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--old", action="store_true")
    ap.add_argument("--mask", default="lv")
    a = ap.parse_args()
    if a.old:
        Hh.checked_step = old_checked_step
    run = Path(a.run)
    args = json.loads((run / "result.json").read_text())["args"]
    with open(run / "skill.pkl", "rb") as f:
        params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    mgr = M.Manager(args["seed"])
    with open(run / "manager.pkl", "rb") as f:
        mgr.params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
    pairs = L.make_split(a.split, a.levels)
    rows, t0 = [], time.time()
    for i, (spec, rank) in enumerate(pairs):
        if a.only is not None and i not in a.only:
            continue
        p = L.payoff(rank)
        g = A.Graph(spec)
        log = []
        ex = Hh.skill_executor(params, spec, p, True, width=args["width"], log=log)
        r = O.run_game(spec, p, A.MASKS[a.mask](spec, mgr.choice(spec, p)), True, ex, BUDGET[a.split], g)
        ticks = sum(x["ticks"] for x in log)
        row = {"i": i, "regret": round(r["regret"], 3), "mistakes": r["mistakes"], "timeout": r["timeout"],
               "dead": r["dead"], "options": len(log), "ticks": ticks,
               "braked": round(sum(x["braked"] for x in log) / max(ticks, 1), 3),
               "fails": [(O.OPT_TYPES[x["otype"]], x["ticks"], x["braked"]) for x in log if not x["ok"]][:6]}
        rows.append(row)
        if row["mistakes"] or row["braked"] > 0.05 or row["timeout"]:
            print(json.dumps(row), flush=True)
    n = len(rows)
    tot = sum(r["ticks"] for r in rows)
    print(json.dumps({"split": a.split, "old": a.old, "n": n, "regret": round(sum(r["regret"] for r in rows) / n, 4),
                      "mistake": sum(r["mistakes"] > 0 for r in rows) / n,
                      "timeout": sum(bool(r["timeout"]) for r in rows) / n,
                      "braked": round(sum(r["braked"] * r["ticks"] for r in rows) / max(tot, 1), 4),
                      "time": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
