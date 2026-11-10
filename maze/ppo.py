"""PPO (PureJaxRL style) with a Gaussian policy, adapted from GRACE-bomber (originally GRACE/craftax/ppo.py).

The actor outputs the mean of a diagonal Gaussian over ``n_actions`` continuous dims (a learned, state-independent
log std, initialised at −0.5); the environment clips actions to [−1, 1].

vmapped envs, ``lax.scan`` rollouts, GAE, clipped surrogate, all jitted; training runs in jitted chunks with a
callback between chunks. An env provides ``obs_dim``, ``n_actions``, ``reset(key) -> (obs, state)`` and
``step(key, state, a) -> (obs, state, reward, done, info)`` with auto-reset inside ``step``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class ActorCritic(nn.Module):
    n_actions: int = 8
    width: int = 512

    @nn.compact
    def __call__(self, x):
        init = nn.initializers.orthogonal(np.sqrt(2))
        h = x
        for _ in range(3):
            h = nn.tanh(nn.Dense(self.width, kernel_init=init)(h))
        mean = nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))(h)
        log_std = self.param("log_std", nn.initializers.constant(-0.5), (self.n_actions,))
        v = x
        for _ in range(3):
            v = nn.tanh(nn.Dense(self.width, kernel_init=init)(v))
        return (mean, jnp.broadcast_to(log_std, mean.shape)), jnp.squeeze(
            nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(v), -1)


def log_prob(pi, a):
    mean, log_std = pi
    return jnp.sum(-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), -1)


def sample(pi, key):
    mean, log_std = pi
    return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)


def entropy(pi):
    return jnp.sum(pi[1] + 0.5 * jnp.log(2 * jnp.pi * jnp.e), -1)


@dataclass
class Config:
    num_envs: int = 1024
    num_steps: int = 64
    lr: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip: float = 0.2
    ent: float = 0.001
    vf: float = 0.5
    epochs: int = 4
    minibatches: int = 8
    max_grad_norm: float = 1.0
    width: int = 512
    total_steps: int = 10_000_000
    chunk_updates: int = 50  # updates per jitted call (evaluation in between)

    @property
    def batch(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def updates(self) -> int:
        return max(1, self.total_steps // self.batch)


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    logp: jax.Array
    obs: jax.Array


def make_train(cfg: Config, env):
    net = ActorCritic(n_actions=env.n_actions, width=cfg.width)
    sched = optax.linear_schedule(cfg.lr, 0.0, cfg.updates * cfg.epochs * cfg.minibatches)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(sched, eps=1e-5))

    def init(key):
        k1, k2 = jax.random.split(key)
        params = net.init(k1, jnp.zeros(env.obs_dim))
        obs, st = jax.vmap(env.reset)(jax.random.split(k2, cfg.num_envs))
        return {"params": params, "opt": tx.init(params), "obs": obs, "st": st, "key": key,
                "stats": jnp.zeros(4)}

    def update(carry, _):
        params, opt, obs, st, key = carry["params"], carry["opt"], carry["obs"], carry["st"], carry["key"]

        def roll(c, _):
            obs, st, key, stats = c
            key, ka, ks = jax.random.split(key, 3)
            pi, v = net.apply(params, obs)
            a = sample(pi, ka)
            logp = log_prob(pi, a)
            obs2, st2, r, d, info = jax.vmap(env.step)(jax.random.split(ks, cfg.num_envs), st, a)
            stats = stats + jnp.array([d.sum(), (r * d).sum(), r.sum(), 0.0])
            return (obs2, st2, key, stats), Transition(d, a, v, r, logp, obs)

        (obs, st, key, stats), tr = jax.lax.scan(roll, (obs, st, key, carry["stats"]), None, cfg.num_steps)
        _, last_v = net.apply(params, obs)

        def gae(c, t):
            adv, nv = c
            delta = t.reward + cfg.gamma * nv * (1 - t.done) - t.value
            adv = delta + cfg.gamma * cfg.gae_lambda * (1 - t.done) * adv
            return (adv, t.value), adv

        _, advs = jax.lax.scan(gae, (jnp.zeros_like(last_v), last_v), tr, reverse=True)
        targets = advs + tr.value

        def loss_fn(p, b, adv, tgt):
            pi, v = net.apply(p, b.obs)
            logp = log_prob(pi, b.action)
            ratio = jnp.exp(logp - b.logp)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            vc = b.value + jnp.clip(v - b.value, -cfg.clip, cfg.clip)
            vl = 0.5 * jnp.maximum((v - tgt) ** 2, (vc - tgt) ** 2).mean()
            ent = entropy(pi).mean()
            return pg + cfg.vf * vl - cfg.ent * ent

        def epoch(c, _):
            params, opt, key = c
            key, kp = jax.random.split(key)
            flat = lambda x: x.reshape((cfg.batch,) + x.shape[2:])  # noqa: E731
            data = (jax.tree_util.tree_map(flat, tr), flat(advs), flat(targets))
            perm = jax.random.permutation(kp, cfg.batch)
            mb = jax.tree_util.tree_map(lambda x: x[perm].reshape((cfg.minibatches, -1) + x.shape[1:]), data)

            def minibatch(c, m):
                params, opt = c
                g = jax.grad(loss_fn)(params, *m)
                upd, opt = tx.update(g, opt, params)
                return (optax.apply_updates(params, upd), opt), None

            (params, opt), _ = jax.lax.scan(minibatch, (params, opt), mb)
            return (params, opt, key), None

        (params, opt, key), _ = jax.lax.scan(epoch, (params, opt, key), None, cfg.epochs)
        return {"params": params, "opt": opt, "obs": obs, "st": st, "key": key, "stats": stats}, None

    @jax.jit
    def chunk(carry):
        carry, _ = jax.lax.scan(update, carry, None, cfg.chunk_updates)
        return carry

    return net, jax.jit(init), chunk


def train(cfg: Config, env, seed: int, callback: Callable | None = None, log=print) -> tuple[Any, list]:
    """Run PPO; ``callback(params, steps) -> dict`` after each chunk. Returns (params, history)."""
    net, init, chunk = make_train(cfg, env)
    carry = init(jax.random.PRNGKey(seed))
    hist, done, t0 = [], 0, time.time()
    n_chunks = -(-cfg.updates // cfg.chunk_updates)
    for c in range(n_chunks):
        carry = chunk(carry)
        jax.block_until_ready(carry["params"])
        done = (c + 1) * cfg.chunk_updates * cfg.batch
        s = np.asarray(carry["stats"])
        row = {"steps": done, "time": time.time() - t0, "episodes": float(s[0]),
               "mean_return": float(s[1] / max(s[0], 1)), "sps": done / (time.time() - t0)}
        carry["stats"] = jnp.zeros(4)
        if callback is not None:
            row.update(callback(carry["params"], done))
        hist.append(row)
        log({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return carry["params"], hist


def policy_fn(n_actions: int = 3, width: int = 512):
    """The policy's mean action (deterministic)."""
    net = ActorCritic(n_actions=n_actions, width=width)
    return jax.jit(lambda params, obs: net.apply(params, obs)[0][0])


__all__ = ["ActorCritic", "Config", "make_train", "train", "policy_fn", "log_prob", "sample", "entropy"]
