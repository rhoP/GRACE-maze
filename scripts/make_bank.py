"""Build the skill bank from oracle episodes on ``mtrain`` levels (``maze/bank.py``).

    PYTHONPATH=. python scripts/make_bank.py --levels 30 --out runs/smoke/bank.pkl            (laptop)
    PYTHONPATH=. python scripts/make_bank.py --levels 1000 --shards 20 --task 3 --out runs/bank  (one cluster shard)
"""

from __future__ import annotations

import argparse
import json
import os

from maze import bank as B
from maze import levels as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=30)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = [i for i in range(a.levels) if i % a.shards == a.task]
    pairs = L.make_split("mtrain", a.levels, indices=idx)
    data = B.collect(pairs, seed=a.task)
    fails = {}
    for f in data["fails"]:
        fails[f[3]] = fails.get(f[3], 0) + 1
    path = a.out if a.shards == 1 and a.out.endswith(".pkl") else os.path.join(a.out, f"bank_{a.task}.pkl")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    B.save(data, path)
    print(json.dumps(B.stats(data) | {"fails": fails, "path": path}))


if __name__ == "__main__":
    main()
