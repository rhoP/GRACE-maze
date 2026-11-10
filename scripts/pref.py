"""Train and evaluate the arms (``docs/plan.md`` §4): skill (PPO on the bank), manager (REINFORCE on the model), flat
A (PPO), and the evaluation.

- abstract model: C, D, D+nr, D+lv (the reference), U, E0, E, E0+nr, E+nr, E+lv, with the manager's agreement with C;
- in the game, learned skill: H+lv (no shield), G+lv (the headline arm), G+nr, and G (no mask);
- in the game, flat: A and A:shield (the same policy under G's braking check at test).

    PYTHONPATH=. python scripts/pref.py --out runs/smoke --bank runs/smoke/bank.pkl --eval-levels 10   (laptop smoke)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from maze import abstract as A  # noqa: E402
from maze import bank as B  # noqa: E402
from maze import envs as E  # noqa: E402
from maze import hier as Hh  # noqa: E402
from maze import levels as L  # noqa: E402
from maze import manager as M  # noqa: E402
from maze import oracle as O  # noqa: E402
from maze import ppo  # noqa: E402

SPLITS = ["mtest", "mpref", "mbig"]
BUDGET = {"mtest": 1500, "mpref": 1500, "mbig": 2500, "mtrain": 1200}  # ticks of 0.1 s
KEYS = ("regret", "mistake", "mistakes", "drop_loss", "timeout_loss", "identity", "success", "timeout", "dead",
        "braked")
ABSTRACT_ARMS = ("C", "D", "D+nr", "D+lv", "U", "E0", "E", "E0+nr", "E+nr", "E+lv")
HIER_ARMS = (("H+lv", False, "lv"), ("G+lv", True, "lv"), ("G+nr", True, "nr"), ("G", True, ""))


def summarise(rs: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rs])) for k in KEYS if k in rs[0]}


def uniform_regret(pairs, n: int = 4) -> float:
    out = []
    for spec, rank in pairs:
        p = L.payoff(rank)
        g = A.Graph(spec)
        out += [A.run_abstract(spec, p, A.uniform_choice(np.random.default_rng(k)), False, 60, g)["regret"]
                for k in range(n)]
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--seed", "--task", type=int, default=0, dest="seed", help="the seed (array task)")
    ap.add_argument("--seed-offset", type=int, default=0, help="seed = task + offset (the final uses 2)")
    ap.add_argument("--skill-steps", type=int, default=2_000_000)
    ap.add_argument("--flat-steps", type=int, default=2_000_000)
    ap.add_argument("--manager-steps", type=int, default=20_000)
    ap.add_argument("--manager-variant", default="nraa", choices=list(M.VARIANTS))
    ap.add_argument("--manager-levels", type=int, default=300)
    ap.add_argument("--flat-levels", type=int, default=300)
    ap.add_argument("--eval-levels", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--arms", nargs="+", default=["abstract", "hier", "flat"])
    ap.add_argument("--resume", action="store_true",
                    help="reuse manager.pkl / skill.pkl in the output folder and keep result.json's other arms")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="manager variants: task t runs variants[t // 2] on seed t % 2 (out/<variant>/seed<k>)")
    a = ap.parse_args()
    out = Path(a.out)
    if a.variants:
        a.manager_variant = a.variants[a.seed // 2]
        a.seed = a.seed % 2
        out = out / a.manager_variant
    a.seed += a.seed_offset
    out = out / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.txt", "a")

    def log(m):
        s = m if isinstance(m, str) else json.dumps(m)
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    log({"args": vars(a)})
    evals = {sp: L.make_split(sp, a.eval_levels) for sp in SPLITS}
    res = {"args": vars(a)}
    if a.resume and (out / "result.json").exists():
        res = json.loads((out / "result.json").read_text()) | {"args": vars(a)}
    for sp, pairs in evals.items():
        res.setdefault(sp, {})["uniform_regret"] = uniform_regret(pairs)

    def save():
        for sp in SPLITS:
            u = res[sp]["uniform_regret"]
            res[sp]["pct_of_uniform"] = {k: round(100 * v["regret"] / u, 1) if u > 0 else None
                                         for k, v in res[sp].items()
                                         if isinstance(v, dict) and "regret" in v}
        (out / "result.json").write_text(json.dumps(res, indent=1))

    # ---------------- manager (abstract) and the abstract arms
    mgr = M.Manager(a.seed)
    if a.resume and (out / "manager.pkl").exists():
        with open(out / "manager.pkl", "rb") as f:
            mgr.params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
        log("manager: loaded")
    elif "abstract" in a.arms or "hier" in a.arms:
        t0 = time.time()
        hist = M.train(mgr, L.make_split("mtrain", a.manager_levels), a.manager_steps, seed=a.seed,
                       eval_every=max(a.manager_steps // 10, 1), evals={k: v[:10] for k, v in evals.items()},
                       log=log, **M.VARIANTS[a.manager_variant])
        res["manager_time"] = time.time() - t0
        (out / "manager_history.json").write_text(json.dumps(hist))
        with open(out / "manager.pkl", "wb") as f:
            pickle.dump(jax.tree_util.tree_map(np.asarray, mgr.params), f)
    if "abstract" in a.arms:
        for sp, pairs in evals.items():
            rows = {k: [] for k in ABSTRACT_ARMS}
            diag = []
            for i, (spec, rank) in enumerate(pairs):
                p = L.payoff(rank)
                g = A.Graph(spec)
                for arm in rows:
                    base = arm.split("+")[0]
                    ch = {"C": A.exact_choice(g, p), "D": A.uniform_choice(np.random.default_rng(i)),
                          "U": A.uniform_choice(np.random.default_rng(i)), "E0": mgr.choice(spec, p),
                          "E": mgr.choice(spec, p)}[base]
                    ch = A.MASKS[arm.partition("+")[2]](spec, ch)
                    rows[arm].append(A.run_abstract(spec, p, ch, base in ("C", "D", "E"), 60, g))
                diag.append(M.agreement(spec, p, g, mgr))
            for arm, rs in rows.items():
                res[sp][arm] = summarise(rs)
            touts = [d for d in diag if d["timeout"]]
            res[sp]["diag"] = {"agree": float(np.mean([x for d in diag for x in d["agree"]])),
                               "loop_share": float(np.mean([d["loop"] for d in touts])) if touts else None}
        save()
        log({sp: res[sp]["pct_of_uniform"] | {"agree": round(res[sp]["diag"]["agree"], 3)} for sp in SPLITS})

    # ---------------- skill and arms H+lv, G+lv, G+nr, G
    if "hier" in a.arms:
        data = B.load(a.bank)
        res["bank"] = B.stats(data)
        log({"bank": res["bank"]})
        env = B.to_env(data, seed=a.seed)
        cfg = ppo.Config(num_envs=a.num_envs, total_steps=a.skill_steps, width=a.width,
                         chunk_updates=max(1, min(20, a.skill_steps // (a.num_envs * 64) // 10)))
        t0 = time.time()
        if a.resume and (out / "skill.pkl").exists():
            with open(out / "skill.pkl", "rb") as f:
                params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
            log("skill: loaded")
        else:
            params, hist = ppo.train(cfg, env, a.seed, log=log)
            res["skill_time"] = time.time() - t0
            (out / "skill_history.json").write_text(json.dumps(hist))
            with open(out / "skill.pkl", "wb") as f:
                pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
        del env, data
        for sp, pairs in evals.items():
            lvs = [L.to_level(spec) for spec, _ in pairs]
            for arm, shield, mask in HIER_ARMS:
                rs, opt_log = [], []
                for i, (spec, rank) in enumerate(pairs):
                    p = L.payoff(rank)
                    g = A.Graph(spec)
                    ex = Hh.skill_executor(params, spec, p, shield, width=a.width, log=opt_log)
                    ch = A.MASKS[mask](spec, mgr.choice(spec, p))
                    rs.append(O.run_game(spec, p, ch, shield, ex, BUDGET[sp], g, lv=lvs[i]))
                res[sp][arm] = summarise(rs)
                by = {}
                for x in opt_log:
                    by.setdefault(O.OPT_TYPES[x["otype"]], []).append(x["ok"])
                ticks = sum(x["ticks"] for x in opt_log)
                res[sp][arm]["delta_by_type"] = {k: round(1 - float(np.mean(v)), 4) for k, v in by.items()}
                res[sp][arm]["delta"] = round(1 - float(np.mean([x["ok"] for x in opt_log])), 4) if opt_log else None
                res[sp][arm]["option_deaths"] = int(sum(x["dead"] for x in opt_log))
                res[sp][arm]["braked"] = round(sum(x["braked"] for x in opt_log) / max(ticks, 1), 4)
                res[sp][arm]["options_per_episode"] = len(opt_log) / len(pairs)
                res[sp][arm]["ticks_per_episode"] = ticks / len(pairs)
            save()
            log({sp: {arm: {k: res[sp][arm][k] for k in ("regret", "mistake", "timeout", "dead", "delta",
                                                          "option_deaths", "braked")} for arm, _, _ in HIER_ARMS}})

    # ---------------- flat A and A:shield
    if "flat" in a.arms:
        specs = L.make_split("mtrain", a.flat_levels)
        bank = E.make_bank([s for s, _ in specs])
        env = E.FlatEnv(bank, jnp.asarray(np.stack([L.payoff(r) for r in L.TRAIN_RANKS])), BUDGET["mtrain"])
        cfg = ppo.Config(num_envs=a.num_envs, total_steps=a.flat_steps, width=a.width,
                         chunk_updates=max(1, min(20, a.flat_steps // (a.num_envs * 64) // 10)))
        t0 = time.time()
        params, hist = ppo.train(cfg, env, a.seed, log=log)
        res["flat_time"] = time.time() - t0
        (out / "flat_history.json").write_text(json.dumps(hist))
        with open(out / "flat.pkl", "wb") as f:
            pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
        for sp, pairs in evals.items():
            for arm, shield in (("A", False), ("A:shield", True)):
                res[sp][arm] = summarise(Hh.eval_flat(params, pairs, shield, BUDGET[sp], width=a.width))
        save()
        log({sp: {arm: res[sp][arm] for arm in ("A", "A:shield")} for sp in SPLITS})

    save()
    log({sp: res[sp]["pct_of_uniform"] for sp in SPLITS})


if __name__ == "__main__":
    main()
