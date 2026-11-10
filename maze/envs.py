"""JAX environments for PPO: the goal-conditioned skill and the flat agent (continuous actions).

Observation (both): a 15 × 15 window of the level raster at 0.25 (± 1.75 units) around the agent with 7 channels
(wall incl. closed fillers, closed door, flap pass direction x / y, key, goal, pit), plus the agent's sub-cell
offset, velocity and keys held.
- Skill: + the waypoint's offset from the agent and the option type (go, open, get).
- Flat: + the payoff (3), the agent's map position, the doors opened, the flaps latched, the items gone, and each
  item's offset and presence (the flat agent knows where the goals and keys are, as the manager does).

Levels are stacked into a Bank (a ``game.Level`` with a leading axis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


from maze import game as G
from maze import levels as L
from maze import sim

V = 15
N_CH = 7
VIEW_DIM = V * V * N_CH
N_SCALARS = 5
N_OPT = 3


def make_bank(specs) -> G.Level:
    lvs = [L.to_level(s) for s in specs]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *lvs)


def level(bank: G.Level, li) -> G.Level:
    return jax.tree_util.tree_map(lambda x: x[li], bank)


def view(lv: G.Level, st: G.State) -> jax.Array:
    pos = st.qpos[:2]
    c = jnp.floor(pos / L.GRES).astype(jnp.int32) + L.GPAD
    w = jax.lax.dynamic_slice(lv.grid, (0, c[1] - V // 2, c[0] - V // 2), (5, V, V)).astype(jnp.int32)
    wall, did, fid, iid, pit = w
    closed = ~G.bits(st.doors, G.ND)
    latched = G.bits(st.flaps, sim.NF)
    dmask = (did > 0) & closed[jnp.maximum(did - 1, 0)]
    fl = fid > 0
    fopen = fl & ~latched[jnp.maximum(fid - 1, 0)]
    wall = (wall > 0) | dmask | (fl & ~fopen)
    fn = lv.flap_n[jnp.maximum(fid - 1, 0)]
    present = ~G.bits(st.gone, G.NI)
    ii = jnp.maximum(iid - 1, 0)
    on = (iid > 0) & present[ii]
    kind = lv.item_kind[ii]
    ch = jnp.stack([wall, dmask, jnp.where(fopen, fn[..., 0], 0.0), jnp.where(fopen, fn[..., 1], 0.0),
                    on & (kind == G.KEY), on & (kind < G.KEY), pit > 0], -1).astype(jnp.float32)
    return ch.reshape(-1)


def scalars(st: G.State) -> jax.Array:
    pos = st.qpos[:2]
    frac = pos / L.GRES - jnp.floor(pos / L.GRES) - 0.5
    return jnp.concatenate([frac, st.qvel[:2] / sim.GEAR, (st.keys / G.KMAX)[None]]).astype(jnp.float32)


# ------------------------------------------------------------------ skill

class SkillState(NamedTuple):
    st: Any
    i: jax.Array
    t: jax.Array


def skill_obs(lv: G.Level, st: G.State, target, otype):
    rel = jnp.clip((target - st.qpos[:2]) / sim.R, -2.0, 2.0)
    return jnp.concatenate([view(lv, st), scalars(st), rel, jax.nn.one_hot(otype, N_OPT)])


SKILL_OBS = VIEW_DIM + N_SCALARS + 2 + N_OPT


def judge(st: G.State, s_abs, t_abs):
    ab = G.abstract_j(st)
    at_s = (ab == s_abs).all()
    at_t = (ab == t_abs).all()
    return at_t & st.alive, ~st.alive | (~at_s & ~at_t)


@dataclass
class SkillEnv:
    bank: G.Level
    starts: Any  # stacked G.State (N,)
    lvl: jax.Array
    s_abs: jax.Array  # (N, 6)
    t_abs: jax.Array
    otype: jax.Array
    target: jax.Array  # (N, 2)
    limit: jax.Array
    n_actions: int = G.N_ACT
    obs_dim: int = SKILL_OBS

    def _draw(self, key):
        i = jax.random.randint(key, (), 0, self.limit.shape[0])
        return SkillState(jax.tree_util.tree_map(lambda x: x[i], self.starts), i, jnp.int32(0))

    def obs(self, s: SkillState):
        return skill_obs(level(self.bank, self.lvl[s.i]), s.st, self.target[s.i], self.otype[s.i])

    def reset(self, key):
        s = self._draw(key)
        return self.obs(s), s

    def step(self, key, s: SkillState, a):
        st = G.step(level(self.bank, self.lvl[s.i]), s.st, a)
        ok, fail = judge(st, self.s_abs[s.i], self.t_abs[s.i])
        done = ok | fail | (s.t + 1 >= self.limit[s.i])
        r = jnp.where(ok, 1.0 - 0.5 * (s.t + 1) / self.limit[s.i], 0.0)
        new = SkillState(st, s.i, s.t + 1)
        nxt = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), self._draw(key), new)
        return self.obs(nxt), nxt, r, done, {"ok": ok}


# ------------------------------------------------------------------ flat

class FlatState(NamedTuple):
    st: Any
    li: jax.Array
    p: jax.Array
    t: jax.Array


def flat_obs(lv: G.Level, st: G.State, p):
    Lm = sim.GM * sim.R
    pos = st.qpos[:2]
    present = ~G.bits(st.gone, G.NI) & (lv.item_kind >= 0)
    off = jnp.where(present[:, None], (lv.item_xy - pos) / Lm, 0.0)
    items = jnp.concatenate([off, present[:, None].astype(jnp.float32)], 1).reshape(-1)
    return jnp.concatenate([view(lv, st), scalars(st), p, pos / Lm, G.bits(st.doors, G.ND).astype(jnp.float32),
                            G.bits(st.flaps, sim.NF).astype(jnp.float32), G.bits(st.gone, G.NI).astype(jnp.float32),
                            items])


FLAT_OBS = VIEW_DIM + N_SCALARS + 3 + 2 + G.ND + sim.NF + G.NI + 3 * G.NI


@dataclass
class FlatEnv:
    bank: G.Level
    payoffs: jax.Array  # (R, 3)
    budget: int
    n_actions: int = G.N_ACT
    obs_dim: int = FLAT_OBS

    def _draw(self, key):
        k1, k2 = jax.random.split(key)
        li = jax.random.randint(k1, (), 0, self.bank.gh.shape[0])
        p = self.payoffs[jax.random.randint(k2, (), 0, self.payoffs.shape[0])]
        return FlatState(G.initial(level(self.bank, li)), li, p, jnp.int32(0))

    def obs(self, s: FlatState):
        return flat_obs(level(self.bank, s.li), s.st, s.p)

    def reset(self, key):
        s = self._draw(key)
        return self.obs(s), s

    def step(self, key, s: FlatState, a):
        st = G.step(level(self.bank, s.li), s.st, a)
        r = jnp.where(st.end >= 0, s.p[jnp.maximum(st.end, 0)], 0.0)
        done = G.done(st) | (s.t + 1 >= self.budget)
        new = FlatState(st, s.li, s.p, s.t + 1)
        nxt = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), self._draw(key), new)
        return self.obs(nxt), nxt, r, done, {"end": st.end, "alive": st.alive}


__all__ = ["make_bank", "level", "SkillEnv", "SkillState", "FlatEnv", "FlatState", "skill_obs", "flat_obs", "judge",
           "view", "SKILL_OBS", "FLAT_OBS", "N_OPT"]
