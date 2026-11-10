"""The learned manager: a scorer over options, trained by REINFORCE on the abstract model (from GRACE-bomber).

Input: a 4 × 4 room map (exists, current room, untaken goals by kind, untaken keys, and the link to the right and
above by state: open, closed door, opened door, flap passable from this room, flap passable into it, latched flap),
the keys held and the payoff; per option: its type, target room, item kind and whether it latches a flap. The logit
of each option comes from convolutions over the map, read at the option's target room.

Training as in GRACE-bomber (``docs/plan.md`` §7 there):
- a one-step exact advantage ψ(s') − ψ(s) − c, with ψ = Φmax − 0.05 · (options to the best goal on the face) and a
  per-option cost c = 0.02;
- a reverse curriculum (an episode starts on the exact manager's path with probability ``curriculum``);
- ``no_revisit`` in the training rollouts, and ``all_actions``: the update is −Σ_a π(a) A(a) with exact advantages.
At test, ``abstract.least_visited`` is the mask (bomber §7).
"""

from __future__ import annotations

import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from maze import abstract as A
from maze import game as G
from maze import levels as L
from maze import sim

CPU = jax.devices("cpu")[0]  # tiny batches: the manager always runs on the CPU
G_MAX = 4
MAXO = 16
N_LINK = 6
CELL = 1 + 1 + 3 + 1 + 2 * N_LINK
MAP_DIM = G_MAX * G_MAX * CELL
GLOB_DIM = (G.KMAX + 1) + 3
OPT_DIM = 3 + G_MAX * G_MAX + 4 + 1
VARIANTS = {"base": {}, "nr": {"no_revisit": True}, "aa": {"all_actions": True},
            "nraa": {"no_revisit": True, "all_actions": True}}


def _cell(spec, rm):
    i, j = divmod(rm, spec.gw)
    return i * G_MAX + j


def link_state(lk, doors, flaps, room):
    if lk.kind == "open":
        return 0
    if lk.kind == "door":
        return 2 if (doors >> lk.door) & 1 else 1
    if (flaps >> lk.flap) & 1:
        return 5
    return 3 if lk.frm == room else 4


def features(spec, s, opts, p):
    room, keys, doors, flaps, gone, end = s
    m = np.zeros((G_MAX * G_MAX, CELL), np.float32)
    for rm in range(spec.n_rooms):
        m[_cell(spec, rm), 0] = 1
    m[_cell(spec, room), 1] = 1
    for ii, (x, y, k, rm) in enumerate(spec.items):
        if not (gone >> ii) & 1:
            m[_cell(spec, rm), 2 + min(k, 3)] = 1
    for lk in spec.links:
        lo = min(lk.a, lk.b)
        h = sim.EDGES[lk.edge][2] == "h"
        m[_cell(spec, lo), 6 + (0 if h else N_LINK) + link_state(lk, doors, flaps, lo)] = 1
    glob = np.concatenate([np.eye(G.KMAX + 1, dtype=np.float32)[min(keys, G.KMAX)], p]).astype(np.float32)
    f = np.zeros((MAXO, OPT_DIM), np.float32)
    mask = np.zeros(MAXO, bool)
    for k, o in enumerate(opts[:MAXO]):
        f[k, A.KINDS.index(o[0])] = 1
        f[k, 3 + _cell(spec, o[2])] = 1
        if o[0] == A.GET:
            f[k, 3 + G_MAX * G_MAX + min(spec.items[o[1]][2], 3)] = 1
        if o[0] == A.GO and spec.links[o[1]].kind == "flap":
            f[k, -1] = 1
        mask[k] = True
    return np.concatenate([m.reshape(-1), glob]), f, mask


class Scorer(nn.Module):
    """Convolutions over the 4 × 4 room grid (with the globals in every cell), then each option is scored from the
    embedding of its target room and of the current room, and its own features."""
    width: int = 128
    layers: int = 6

    @nn.compact
    def __call__(self, g, f, mask):
        m = g[..., :MAP_DIM].reshape(g.shape[:-1] + (G_MAX, G_MAX, CELL))
        glob = g[..., MAP_DIM:]
        x = jnp.concatenate([m, jnp.broadcast_to(glob[..., None, None, :], m.shape[:-1] + glob.shape[-1:])], -1)
        x = nn.relu(nn.Conv(self.width, (1, 1))(x))
        for _ in range(self.layers):
            x = x + nn.relu(nn.Conv(self.width, (3, 3), padding="SAME")(x))
        emb = x.reshape(x.shape[:-3] + (G_MAX * G_MAX, self.width))
        cur = jnp.einsum("...c,...cw->...w", m[..., 1].reshape(m.shape[:-3] + (G_MAX * G_MAX,)), emb)
        tgt = jnp.einsum("...oc,...cw->...ow", f[..., 3:3 + G_MAX * G_MAX], emb)
        h = jnp.concatenate([tgt, jnp.broadcast_to(cur[..., None, :], tgt.shape), f], -1)
        h = nn.relu(nn.Dense(self.width)(h))
        logit = nn.Dense(1)(h)[..., 0]
        return jnp.where(mask, logit, -1e9)


class Manager:
    def __init__(self, seed: int = 0, lr: float = 1e-3, ent: float = 0.01, width: int = 128):
        self.net = Scorer(width)
        self.tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
        with jax.default_device(CPU):
            self.params = self.net.init(jax.random.PRNGKey(seed), jnp.zeros(MAP_DIM + GLOB_DIM),
                                        jnp.zeros((MAXO, OPT_DIM)), jnp.ones(MAXO, bool))
            self.opt = self.tx.init(self.params)
        self.ent = ent
        self._logits = jax.jit(self.net.apply)

        def entropy(lp, mask):
            return -jnp.sum(jnp.where(mask, jnp.exp(lp) * lp, 0.0), -1)

        def loss(params, g, f, mask, a, adv):
            lp = jax.nn.log_softmax(self.net.apply(params, g, f, mask))
            pa = jnp.take_along_axis(lp, a[:, None], 1)[:, 0]
            return -(adv * pa).mean() - self.ent * entropy(lp, mask).mean()

        def loss_all(params, g, f, mask, a, adv):  # adv: (batch, MAXO)
            lp = jax.nn.log_softmax(self.net.apply(params, g, f, mask))
            pr = jnp.where(mask, jnp.exp(lp), 0.0)
            return -jnp.sum(pr * adv, -1).mean() - self.ent * entropy(lp, mask).mean()

        def make_update(fn):
            def update(params, opt, *batch):
                g = jax.grad(fn)(params, *batch)
                u, opt = self.tx.update(g, opt, params)
                return optax.apply_updates(params, u), opt
            return jax.jit(update)

        self._update_fns = {False: make_update(loss), True: make_update(loss_all)}

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = jax.device_put(value, CPU)

    def _update(self, params, opt, *batch, all_actions: bool = False):
        with jax.default_device(CPU):
            return self._update_fns[all_actions](params, opt, *(jax.device_put(b, CPU) for b in batch))

    def choice(self, spec, p, rng=None, greedy: bool = True, record: list | None = None):
        def choose(s, opts, ks):
            g, f, mask = features(spec, s, opts, p)
            allow = np.zeros(MAXO, bool)
            allow[[k for k in ks if k < MAXO]] = True
            lg = np.asarray(self._logits(self.params, *jax.device_put((g, f, allow), CPU)))
            if greedy:
                k = int(np.argmax(lg))
            else:
                pr = np.exp(lg - lg.max())
                pr /= pr.sum()
                k = int(rng.choice(MAXO, p=pr))
            if record is not None:
                record.append((g, f, allow, k, s))
            return k
        return choose


def exact_path(spec, p, g: A.Graph, max_options: int = 60) -> list:
    s, path = A.start(spec), []
    choose = A.exact_choice(g, p)
    while s[A.END] < 0 and len(path) < max_options:
        opts = A.options(spec, s)
        if not opts:
            break
        path.append(s)
        s = A.succ(spec, s, opts[choose(s, opts, A.allowed(g, s, opts, p))])
    return path


def evaluate(mgr: Manager, pairs, shield: bool, max_options: int = 60, no_revisit: bool = False) -> dict:
    rs = []
    for spec, rank in pairs:
        p = L.payoff(rank)
        g = A.Graph(spec)
        ch = mgr.choice(spec, p)
        rs.append(A.run_abstract(spec, p, A.no_revisit(spec, ch) if no_revisit else ch, shield, max_options, g))
    return {k: float(np.mean([r[k] for r in rs])) for k in ("regret", "mistake", "timeout", "identity")}


def psi(g: A.Graph, s, p, c: float) -> float:
    """Φmax − c · options to the nearest best goal on the face (the payoff itself once an exit is taken)."""
    if s[A.END] >= 0:
        return float(p[s[A.END]])
    v = g.phi(s, p)
    best = [k for k in g.face[s] if abs(float(p[k]) - v) < 1e-9]
    d = min((g.dist[k].get(s, 0) for k in best), default=0)
    return v - c * d


def agreement(spec, p, g: A.Graph, mgr: Manager, max_options: int = 60) -> dict:
    """One plain greedy episode: whether an abstract state repeats, and per decision whether the choice's successor is
    as close to the best goal on the face as C's."""
    choose = mgr.choice(spec, p)
    s, seen, n, agree, loop = A.start(spec), set(), 0, [], False
    while s[A.END] < 0 and n < max_options:
        opts = A.options(spec, s)
        if not opts:
            break
        loop |= s in seen
        seen.add(s)
        v = g.phi(s, p)
        best = [k for k in g.face[s] if abs(float(p[k]) - v) < 1e-9]

        def cost(i):
            t = A.succ(spec, s, opts[i])
            g.ensure(t)
            if t[A.END] >= 0:
                return 0 if t[A.END] in best else 10 ** 6
            return min((g.dist[k].get(t, 10 ** 6) for k in best), default=10 ** 6)
        k = choose(s, opts, list(range(len(opts))))
        agree.append(cost(k) <= min(cost(i) for i in range(len(opts))))
        s = A.succ(spec, s, opts[k])
        n += 1
    return {"timeout": s[A.END] < 0, "loop": loop, "agree": agree}


def train(mgr: Manager, pairs, steps: int, seed: int = 0, batch: int = 32, curriculum: float = 0.9,
          dist_cost: float = 0.05, step_cost: float = 0.02, max_options: int = 60, eval_every: int = 50_000,
          evals: dict | None = None, no_revisit: bool = False, all_actions: bool = False, log=print) -> list:
    """REINFORCE with the td advantage; ``steps`` counts decisions. Eval rows: the greedy regret on each split plain
    (``<split>``) and with the mask and the shield (``<split>:nr``)."""
    rng = np.random.default_rng(seed)
    graphs = {}
    hist, done, t0, next_eval = [], 0, time.time(), 0
    while done < steps:
        if evals and done >= next_eval:
            row = {"steps": done, "time": time.time() - t0}
            for name, ev in evals.items():
                row[name] = evaluate(mgr, ev, shield=False)["regret"]
                row[name + ":nr"] = evaluate(mgr, ev, shield=True, no_revisit=True)["regret"]
            hist.append(row)
            log(row)
            next_eval += eval_every
        Gs, F, M, Aa, ADV = [], [], [], [], []
        for _ in range(batch):
            i = int(rng.integers(len(pairs)))
            spec, rank = pairs[i]
            p = L.payoff(rank)
            if i not in graphs:
                graphs[i] = (A.Graph(spec), None)
            g = graphs[i][0]
            s0 = A.start(spec)
            if rng.random() < curriculum:
                if graphs[i][1] is None:
                    graphs[i] = (g, exact_path(spec, p, g))
                path = graphs[i][1]
                if path:
                    s0 = path[int(rng.integers(len(path)))]
            rec = []
            choose = mgr.choice(spec, p, rng, greedy=False, record=rec)
            if no_revisit:
                choose = A.no_revisit(spec, choose)
            s, n = s0, 0
            while s[A.END] < 0 and n < max_options:
                opts = A.options(spec, s)
                if not opts:
                    break
                s = A.succ(spec, s, opts[choose(s, opts, list(range(len(opts))))])
                n += 1
            phis = [psi(g, r[4], p, dist_cost) for r in rec]
            nxt = phis[1:] + [psi(g, s, p, dist_cost)]
            for r, a_, b_ in zip(rec, phis, nxt):
                Gs.append(r[0]); F.append(r[1]); M.append(r[2]); Aa.append(r[3])
                if all_actions:
                    adv = np.zeros(MAXO, np.float32)
                    for j, o in enumerate(A.options(spec, r[4])[:MAXO]):
                        if r[2][j]:
                            t = A.succ(spec, r[4], o)
                            g.ensure(t)
                            adv[j] = psi(g, t, p, dist_cost) - a_ - step_cost
                    ADV.append(adv)
                else:
                    ADV.append(b_ - a_ - step_cost)
            done += len(rec)
        n = len(Aa)
        b = 1 << max(n - 1, 1).bit_length()
        Gs += [Gs[0]] * (b - n); F += [F[0]] * (b - n); M += [np.zeros(MAXO, bool)] * (b - n)
        Aa += [0] * (b - n); ADV += [np.zeros(MAXO, np.float32) if all_actions else 0.0] * (b - n)
        mgr.params, mgr.opt = mgr._update(mgr.params, mgr.opt, jnp.asarray(np.stack(Gs)), jnp.asarray(np.stack(F)),
                                          jnp.asarray(np.stack(M)), jnp.asarray(np.array(Aa)),
                                          jnp.asarray(np.array(ADV, np.float32)), all_actions=all_actions)
    return hist


__all__ = ["Manager", "features", "train", "evaluate", "exact_path", "agreement", "VARIANTS"]
