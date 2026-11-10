"""The skill bank: option starts from oracle rollouts on the training levels (adapted from GRACE-bomber).

Episodes run through the game with the oracle executor, under the exact manager C and the uniform shielded manager D
(for variety). Every successful oracle option gives one entry at its start, plus ``mid`` entries at random points
along its trajectory (a reverse curriculum). Entries are balanced by option type when the env is built.
"""

from __future__ import annotations

import glob
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from maze import abstract as A
from maze import envs as E
from maze import game as G
from maze import levels as L
from maze import oracle as O


def limit_for(n: int) -> int:
    return 2 * n + 40


def collect(pairs, seed: int = 0, mid: int = 2, episodes=("C", "D"), budget: int = 1500, log=print) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for li, (spec, rank) in enumerate(pairs):
        p = L.payoff(rank)
        g = A.Graph(spec)
        lv_li = L.to_level(spec)
        for arm in episodes:
            choose = A.exact_choice(g, p) if arm == "C" else A.uniform_choice(rng)

            def executor(spec_, lv, st, s, o):
                r = O.run_option(spec_, lv, st, s, o)
                if r is None:
                    return st, 0, []
                fin, n, ok, traj = r
                if ok and n > 0:
                    base = dict(lvl=li, s_abs=s, t_abs=A.succ(spec_, s, o), otype=O.option_type(o),
                                target=O.waypoint(spec_, s, o), n=n)
                    rows.append(dict(base, st=G.to_np(st), limit=limit_for(n), offset=0))
                    for k in rng.integers(1, max(n, 2), size=min(mid, max(n - 1, 0))):
                        rows.append(dict(base, st=G.to_np(jax.tree_util.tree_map(lambda x: x[k - 1], traj)),
                                         limit=limit_for(n - k), offset=int(k)))
                return fin, n, []

            O.run_game(spec, p, choose, True, executor, budget, g, lv=lv_li)
        if (li + 1) % 10 == 0:
            log(f"bank: {li + 1}/{len(pairs)} levels, {len(rows)} entries, {len(O.FAILS)} failed plans, "
                f"{time.time() - t0:.0f}s")
    return {"rows": rows, "specs": [s for s, _ in pairs], "fails": list(O.FAILS)}


def to_env(data: dict, balance: bool = True, seed: int = 0, n: int | None = None) -> E.SkillEnv:
    rows = data["rows"]
    if balance:
        rng = np.random.default_rng(seed)
        by = {}
        for i, r in enumerate(rows):
            by.setdefault(r["otype"], []).append(i)
        n = n or len(rows)
        per = n // len(by)
        sel = np.concatenate([rng.choice(v, size=per, replace=len(v) < per) for v in by.values()])
        rows = [rows[i] for i in sel]
    bank = E.make_bank(data["specs"])
    starts = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[r["st"] for r in rows])
    arr = lambda k, d=np.int32: jnp.asarray(np.array([r[k] for r in rows], d))  # noqa: E731
    return E.SkillEnv(bank, starts, arr("lvl"), arr("s_abs"), arr("t_abs"), arr("otype"), arr("target", np.float32),
                      arr("limit"))


def stats(data: dict) -> dict:
    rows = [r for r in data["rows"] if r["offset"] == 0]
    out = {}
    for t, name in enumerate(O.OPT_TYPES):
        ns = [r["n"] for r in rows if r["otype"] == t]
        out[name] = {"options": len(ns), "ticks_mean": float(np.mean(ns)) if ns else 0.0,
                     "ticks_p95": float(np.percentile(ns, 95)) if ns else 0.0}
    out["entries"] = len(data["rows"])
    return out


def save(data: dict, path):
    rows = [dict(r, st=jax.tree_util.tree_map(np.asarray, r["st"])) for r in data["rows"]]
    with open(path, "wb") as f:
        pickle.dump({"rows": rows, "specs": data["specs"], "fails": data.get("fails", [])}, f)


def load(path) -> dict:
    """One bank file, or every ``bank_*.pkl`` in a folder (merged: level indices are offset)."""
    files = sorted(glob.glob(os.path.join(path, "bank_*.pkl"))) if os.path.isdir(path) else [path]
    rows, specs = [], []
    for fn in files:
        with open(fn, "rb") as f:
            d = pickle.load(f)
        for r in d["rows"]:
            r["st"] = jax.tree_util.tree_map(jnp.asarray, r["st"])
            r["lvl"] += len(specs)
            rows.append(r)
        specs += d["specs"]
    return {"rows": rows, "specs": specs}


__all__ = ["collect", "to_env", "stats", "save", "load", "limit_for"]
