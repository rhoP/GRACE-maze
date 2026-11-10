"""Core tests: standalone, physics of flaps and doors, the braking bound, the exact manager and the oracle."""

from __future__ import annotations

import ast
import glob

import jax
import jax.numpy as jnp
import numpy as np

from maze import abstract as A
from maze import game as G
from maze import hier as Hh
from maze import levels as L
from maze import oracle as O
from maze import sim


def test_standalone():
    for f in glob.glob("maze/*.py") + glob.glob("scripts/*.py"):
        for node in ast.walk(ast.parse(open(f).read())):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any(n.split(".")[0] in ("GRACE", "bomb", "plat") for n in names), (f, names)


def _level_with(kind):
    for spec, _ in L.make_split("mtrain", 20):
        if getattr(spec, kind):
            return spec
    raise AssertionError(kind)


def _place(lv, st, xy, room):
    return st._replace(qpos=st.qpos.at[:2].set(jnp.asarray(xy, jnp.float32)), room=jnp.int32(room))


def _drive(lv, st, u, n, use=0.0):
    step = jax.jit(G.step)
    for _ in range(n):
        st = step(lv, st, jnp.array([u[0], u[1], use], jnp.float32))
    return st


def test_flap_one_way_and_latches():
    spec = _level_with("flaps")
    lk = spec.flaps[0]
    lv = L.to_level(spec)
    c = sim.gap_centre(lk.edge)
    n = sim.normal(lk.edge, spec.frame_room(lk.to) == sim.EDGES[lk.edge][1])
    st = _place(lv, G.initial(lv), c - n, lk.frm)
    st = _drive(lv, st, n, 30)
    assert int(st.room) == lk.to and (int(st.flaps) >> lk.flap) & 1
    back = _drive(lv, st, -n, 60)
    assert int(back.room) == lk.to
    st = _place(lv, G.initial(lv), c + n, lk.to)
    st = _drive(lv, st, -n, 60)
    assert int(st.room) == lk.to and int(st.flaps) == 0


def test_door_needs_key():
    spec = _level_with("doors")
    lk = spec.doors[0]
    lv = L.to_level(spec)
    c = sim.gap_centre(lk.edge)
    n = sim.normal(lk.edge, True)
    st = _place(lv, G.initial(lv), c - n, lk.a)._replace(keys=jnp.int32(0))
    assert int(_drive(lv, st, n, 40, use=1.0).room) == lk.a
    st = st._replace(keys=jnp.int32(2))
    st = _drive(lv, st, n, 40, use=1.0)
    assert int(st.room) == lk.b and int(st.keys) == 1 and int(st.doors) == 1 << lk.door


def test_braking_bound():
    spec, _ = L.make_split("mtrain", 1)[0]
    lv = L.to_level(spec)
    rng = np.random.default_rng(0)
    step = jax.jit(G.step)
    worst = 0.0
    for _ in range(40):
        v = rng.uniform(-2, 2, 2)
        st = _place(lv, G.initial(lv), spec.spawn, spec.spawn_room)
        st = st._replace(qvel=st.qvel.at[:2].set(jnp.asarray(v, jnp.float32)))
        bound = np.asarray(Hh.stop_dist(jnp.asarray(v)))
        p0 = np.asarray(st.qpos[:2])
        for _ in range(40):
            st = step(lv, st, Hh.brake_action(st))
        d = np.asarray(st.qpos[:2]) - p0
        worst = max(worst, float(np.max(np.abs(d) - np.abs(bound))))
    assert worst <= Hh.MARGIN, worst


def test_exact_manager_identity_and_oracle():
    for split in L.SPLITS:
        for spec, rank in L.make_split(split, 4):
            p = L.payoff(rank)
            g = A.Graph(spec)
            r = A.run_abstract(spec, p, A.exact_choice(g, p), True, 60, g)
            assert r["regret"] == 0 and r["mistakes"] == 0 and r["identity"] == 1
            u = A.run_abstract(spec, p, A.uniform_choice(np.random.default_rng(0)), False, 60, g)
            assert u["identity"] == 1
            for mask in (A.no_revisit, A.least_visited):
                n = A.run_abstract(spec, p, mask(spec, A.uniform_choice(np.random.default_rng(0))), True, 60, g)
                assert n["mistakes"] == 0 and n["identity"] == 1
    spec, rank = L.make_split("mpref", 1)[0]
    p = L.payoff(rank)
    g = A.Graph(spec)
    r = O.run_game(spec, p, A.exact_choice(g, p), True, O.oracle_executor, 3000, g)
    assert r["regret"] == 0 and r["identity"] == 1 and r["dead"] == 0


def test_check_brakes_before_a_pit():
    for spec, rank in L.make_split("mtrain", 20):
        if spec.pits:
            break
    lv = L.to_level(spec)
    table = jnp.asarray(A.face_table(spec))
    p = jnp.asarray(L.payoff(rank))
    x0, y0, x1, y1 = spec.pits[0]
    cy = (y0 + y1) / 2
    room = int(G.room_of(lv, jnp.array([x0, cy])))
    st = _place(lv, G.initial(lv), (x0 - 0.9, cy), room)._replace(qvel=jnp.zeros(2 + sim.NF).at[0].set(1.8))
    step = jax.jit(lambda s: Hh.checked_step(lv, table, p, s, jnp.array([1.0, 0.0, -1.0])))
    braked = 0
    for _ in range(30):
        st, b = step(st)
        braked += int(b)
        assert bool(st.alive)
    assert braked > 0
    free = jax.jit(G.step)
    st = _place(lv, G.initial(lv), (x0 - 0.9, cy), room)._replace(qvel=jnp.zeros(2 + sim.NF).at[0].set(1.8))
    for _ in range(30):
        st = free(lv, st, jnp.array([1.0, 0.0, -1.0]))
    assert not bool(st.alive)
