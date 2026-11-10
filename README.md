# GRACE-maze

GRACE's hierarchy (exact abstract layer, option shield, learned manager and skills) on a MuJoCo (MJX) point-mass
maze with one-way latching flaps, key doors and pits. See `docs/plan.md`.

    PYTHONPATH=. python -m pytest -q tests
    PYTHONPATH=. python scripts/gates.py --p1 --p2 --p1-levels 5
    PYTHONPATH=. python scripts/make_bank.py --levels 30 --out runs/smoke/bank.pkl
    PYTHONPATH=. python scripts/pref.py --out runs/smoke --bank runs/smoke/bank.pkl --eval-levels 10
    PYTHONPATH=. python scripts/figures.py --out plots [--run runs/pilot/seed0]
    bash scripts/cluster/submit.sh p1|bank|pilot|mgr|final
