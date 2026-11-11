"""The game: MJX physics (``sim``) plus the discrete logic, jitted and batched over levels.

Action: (fx, fy, use) in [-1, 1]³; the force goes to the point mass, and ``use`` > 0.5 opens a closed door within USE
of the agent's centre when a key is held (the key is spent). Events after the physics step:
- the agent's centre enters a pit: dead;
- the centre comes within PICK of a present item: a key adds a key (at most KMAX); a goal ends the episode;
- the centre crosses from a flap's *from* room into its *to* room: the flap latches (its filler closes the gap).

Abstract state (``abstract_j``): (room, keys, doors opened, flaps latched, items gone, end). A level is stored as a
``Level`` of arrays (``levels.to_level``); stacked Levels (leading axis) are vmapped.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from maze import sim

ND = 4  # doors
NI = 5  # items: goals A, B, C at indices 0-2, keys at 3-4
NP = 16  # pits
KMAX = 3
KEY = 3  # item kind of a key
PICK = 0.35
USE = 0.7
N_ACT = 3


class Level(NamedTuple):
    gh: jax.Array
    gw: jax.Array
    geom_base: jax.Array  # (NGEOM, 3) fillers at their gaps
    body_pos: jax.Array
    body_quat: jax.Array
    fill_kind: jax.Array  # (NEDGE,) 0 wall, 1 open, 2 door, 3 flap
    fill_idx: jax.Array  # (NEDGE,) door or flap index
    door_xy: jax.Array  # (ND, 2)
    door_ok: jax.Array  # (ND,)
    flap_from: jax.Array  # (NF,) level rooms
    flap_to: jax.Array
    flap_ok: jax.Array
    flap_xy: jax.Array  # (NF, 2) gap centre
    flap_n: jax.Array  # (NF, 2) pass direction
    item_xy: jax.Array  # (NI, 2)
    item_kind: jax.Array  # (NI,) -1 none
    item_room: jax.Array
    pit: jax.Array  # (NP, 4) x0, y0, x1, y1
    spawn: jax.Array  # (2,)
    keys0: jax.Array
    grid: jax.Array  # (5, GP, GP) int8 raster for observations (envs.py)


class State(NamedTuple):
    qpos: jax.Array  # (2 + NF,)
    qvel: jax.Array
    t: jax.Array
    room: jax.Array
    keys: jax.Array
    doors: jax.Array
    flaps: jax.Array
    gone: jax.Array
    end: jax.Array
    alive: jax.Array


def room_of(lv: Level, xy):
    i = jnp.clip(jnp.floor(xy[1] / sim.R).astype(jnp.int32), 0, lv.gh - 1)
    j = jnp.clip(jnp.floor(xy[0] / sim.R).astype(jnp.int32), 0, lv.gw - 1)
    return i * lv.gw + j


def initial(lv: Level) -> State:
    q = jnp.zeros(2 + sim.NF).at[:2].set(lv.spawn)
    i32 = jnp.int32
    return State(q, jnp.zeros(2 + sim.NF), i32(0), room_of(lv, lv.spawn), lv.keys0.astype(i32), i32(0), i32(0),
                 i32(0), i32(-1), jnp.asarray(True))


def bits(x, n):
    return ((x >> jnp.arange(n)) & 1) > 0


def fillers_present(lv: Level, doors, flaps):
    k, i = lv.fill_kind, jnp.maximum(lv.fill_idx, 0)
    return (k == 0) | ((k == 2) & ~bits(doors, ND)[jnp.minimum(i, ND - 1)]) | ((k == 3) & bits(flaps, sim.NF)[i])


def level_mx(lv: Level, doors, flaps):
    _, mx, _, ids = sim.model()
    fid = jnp.asarray(ids["fill"])
    present = fillers_present(lv, doors, flaps)
    gp = lv.geom_base.at[fid].set(jnp.where(present[:, None], lv.geom_base[fid], jnp.asarray(sim.FAR)))
    return sim.level_model(mx, gp, lv.body_pos, lv.body_quat)


def try_use(lv: Level, st: State, use):
    pos = st.qpos[:2]
    closed = ~bits(st.doors, ND) & lv.door_ok
    near = closed & (jnp.linalg.norm(lv.door_xy - pos, axis=1) < USE)
    can = use & (st.keys > 0) & near.any() & st.alive & (st.end < 0)
    first = jnp.argmax(near)
    return jnp.where(can, st.doors | (1 << first), st.doors), st.keys - can.astype(jnp.int32)


def step(lv: Level, st: State, a) -> State:
    a = jnp.clip(a, -1.0, 1.0)
    live = st.alive & (st.end < 0)
    doors, keys = try_use(lv, st, a[2] > 0.5)
    _, _, d0, _ = sim.model()
    qpos, qvel = sim.physics(level_mx(lv, doors, st.flaps), d0, st.qpos, st.qvel, a[:2] * live)
    qpos = jnp.where(live, qpos, st.qpos)
    qvel = jnp.where(live, qvel, jnp.zeros_like(qvel))
    pos = qpos[:2]
    room = room_of(lv, pos)
    cross = lv.flap_ok & ~bits(st.flaps, sim.NF) & (lv.flap_from == st.room) & (lv.flap_to == room) & (room != st.room)
    flaps = st.flaps | jnp.sum(cross.astype(jnp.int32) << jnp.arange(sim.NF))
    # The latched filler closes the gap around an agent whose centre is just past the centreline. Contact then
    # pushed it on at up to 3 units/s (pilot, mpref 35: above the top speed of 2, into an item). Instead the agent
    # is placed clear of the filler along the flap's pass direction, with its velocity unchanged.
    need = jnp.maximum(0.0, sim.T + sim.RAD + 0.01 - jnp.sum((pos - lv.flap_xy) * lv.flap_n, axis=1))
    pos = pos + jnp.sum(jnp.where(cross[:, None], lv.flap_n * need[:, None], 0.0), axis=0)
    qpos = qpos.at[:2].set(pos)
    # A latched flap's blade is held shut inside its filler. Left free, it sprang shut through the to room and swept
    # the agent along (final seed 2, mpref 23: into the cheap goal against full braking).
    shut = bits(flaps, sim.NF)
    qpos = qpos.at[2:].set(jnp.where(shut, 0.0, qpos[2:]))
    qvel = qvel.at[2:].set(jnp.where(shut, 0.0, qvel[2:]))
    in_pit = ((pos[0] > lv.pit[:, 0]) & (pos[0] < lv.pit[:, 2]) & (pos[1] > lv.pit[:, 1]) & (pos[1] < lv.pit[:, 3]))
    alive = st.alive & ~(live & in_pit.any())
    present = ~bits(st.gone, NI) & (lv.item_kind >= 0)
    got = present & (jnp.linalg.norm(lv.item_xy - pos, axis=1) < PICK) & live & alive
    gone = st.gone | jnp.sum(got.astype(jnp.int32) << jnp.arange(NI))
    keys = jnp.minimum(keys + jnp.sum(got & (lv.item_kind == KEY)), KMAX)
    goal = got & (lv.item_kind < KEY)
    end = jnp.where((st.end < 0) & goal.any(), lv.item_kind[jnp.argmax(goal)], st.end)
    return State(qpos, qvel, st.t + 1, room, keys.astype(jnp.int32), doors, flaps, gone, end.astype(jnp.int32), alive)


def abstract_j(st: State):
    return jnp.stack([st.room, st.keys, st.doors, st.flaps, st.gone, st.end])


def done(st: State):
    return ~st.alive | (st.end >= 0)


def to_np(st: State) -> State:
    return jax.tree_util.tree_map(np.asarray, st)


__all__ = ["Level", "State", "initial", "step", "abstract_j", "done", "room_of", "bits", "fillers_present",
           "ND", "NI", "NP", "KMAX", "KEY", "PICK", "USE", "N_ACT"]
