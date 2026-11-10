"""MuJoCo (MJX) model of the maze frame: a point mass among walls, gap fillers and hinged flaps.

The frame is GM × GM rooms of R × R units (x right, y up; room (i, j) spans x ∈ [jR, (j+1)R], y ∈ [iR, (i+1)R]).
Every edge between two frame rooms has two static wall segments and a gap of width GAP in the middle. The gap holds a
*filler* box (a wall, a closed door, or a latched flap) or nothing (an open link, an opened door, an unlatched flap).

One compiled model serves every level: per-level geometry goes into the batched model fields ``geom_pos`` (fillers
present, or parked far away) and ``body_pos`` / ``body_quat`` (flap hinges, or parked). Mocap bodies are not used:
MJX does not collide the jointed children of a mocap body (checked 2026-11-10).

A flap is a box on a hinge at one end of its gap, with joint range [0, 100°]: pushed from its *from* side it swings
open; pushed from the other side it rests on its joint limit. An open flap also lets the agent back through, so the
game latches it (its filler appears) the moment the agent's centre crosses into the *to* room (``game.step``).
"""

from __future__ import annotations

from functools import lru_cache

import jax
import mujoco
import numpy as np
from mujoco import mjx

R = 4.0  # room size
T = 0.1  # half wall thickness
GAP = 1.2  # gap width
GM = 4  # frame rooms per side
RAD = 0.2  # agent radius
GEAR = 2.0  # max force per axis
DAMP = 1.0  # velocity damping (top speed GEAR / DAMP)
MASS = 1.0
DT = 0.02
SUB = 5  # physics steps per control step (0.1 s)
NF = 4  # flap bodies
ITER, LS_ITER = 4, 8  # solver iterations (MJX runs the batch until every env converges)
FAR = np.array([100.0, 100.0, 0.0])


def _edges():
    out = []  # (a, b, axis): frame rooms; "h": a left of b (wall x = const), "v": a below b (wall y = const)
    for i in range(GM):
        for j in range(GM - 1):
            out.append((i * GM + j, i * GM + j + 1, "h"))
    for i in range(GM - 1):
        for j in range(GM):
            out.append((i * GM + j, (i + 1) * GM + j, "v"))
    return out


EDGES = _edges()
NEDGE = len(EDGES)
EDGE_OF = {(a, b): k for k, (a, b, _) in enumerate(EDGES)} | {(b, a): k for k, (a, b, _) in enumerate(EDGES)}


def gap_centre(k: int) -> np.ndarray:
    a, b, ax = EDGES[k]
    i, j = divmod(a, GM)
    if ax == "h":
        return np.array([(j + 1) * R, i * R + R / 2])
    return np.array([j * R + R / 2, (i + 1) * R])


def normal(k: int, to_b: bool = True) -> np.ndarray:
    """Unit normal of edge k's wall pointing into room b (or a)."""
    n = np.array([1.0, 0.0]) if EDGES[k][2] == "h" else np.array([0.0, 1.0])
    return n if to_b else -n


def _xml() -> str:
    L = GM * R
    g = []
    box = lambda name, x, y, a, b: g.append(  # noqa: E731
        f'<geom name="{name}" type="box" pos="{x} {y} 0" size="{a} {b} 0.25" contype="2" conaffinity="1" '
        f'rgba=".3 .3 .35 1"/>')
    box("bd0", L / 2, 0, L / 2 + T, T)
    box("bd1", L / 2, L, L / 2 + T, T)
    box("bd2", 0, L / 2, T, L / 2 + T)
    box("bd3", L, L / 2, T, L / 2 + T)
    side = (R - GAP) / 4  # half length of a side segment
    for k, (a, b, ax) in enumerate(EDGES):
        c = gap_centre(k)
        off = GAP / 2 + side
        for s, sg in enumerate((-1, 1)):
            if ax == "h":
                box(f"s{k}_{s}", c[0], c[1] + sg * off, T, side + T)
            else:
                box(f"s{k}_{s}", c[0] + sg * off, c[1], side + T, T)
    for k, (a, b, ax) in enumerate(EDGES):
        hx, hy = (T, GAP / 2 + 0.02) if ax == "h" else (GAP / 2 + 0.02, T)
        box(f"fill{k}", FAR[0], FAR[1], hx, hy)
    flaps = "".join(
        f'<body name="flap{f}" pos="{200 + 5 * f} 200 0"><joint name="hinge{f}" type="hinge" axis="0 0 1" '
        f'range="0 100" limited="true" stiffness="0.5" damping="0.05"/><geom name="blade{f}" type="box" '
        f'pos="{GAP / 2} 0 0" size="{GAP / 2 - 0.02} 0.05 0.25" mass="0.1" contype="2" conaffinity="1" '
        f'rgba=".75 .55 .2 1"/></body>' for f in range(NF))
    return f'''<mujoco model="grace_maze"><compiler angle="degree"/>
<option timestep="{DT}" integrator="implicitfast" gravity="0 0 0" iterations="{ITER}" ls_iterations="{LS_ITER}"/>
<default><geom condim="1"/></default>
<worldbody>{"".join(g)}
<body name="agent" pos="0 0 0"><joint name="ax" type="slide" axis="1 0 0" damping="{DAMP}"/>
<joint name="ay" type="slide" axis="0 1 0" damping="{DAMP}"/>
<geom name="agent" type="sphere" size="{RAD}" mass="{MASS}" contype="1" conaffinity="2" rgba=".2 .7 .35 1"/></body>
{flaps}
</worldbody><actuator><motor joint="ax" gear="{GEAR}" ctrlrange="-1 1"/><motor joint="ay" gear="{GEAR}" ctrlrange="-1 1"/>
</actuator></mujoco>'''


@lru_cache(maxsize=None)
def model():
    """(MjModel, mjx.Model, mjx.Data template, ids)."""
    m = mujoco.MjModel.from_xml_string(_xml())
    mx = mjx.put_model(m)
    d0 = mjx.make_data(m)
    ids = {"fill": np.array([m.geom(f"fill{k}").id for k in range(NEDGE)]),
           "flap_body": np.array([m.body(f"flap{f}").id for f in range(NF)])}
    return m, mx, d0, ids


def flap_pose(k: int, to_b: bool) -> tuple[np.ndarray, np.ndarray]:
    """Hinge position and orientation for a flap in edge k that opens toward room b (``to_b``) or a.

    In the flap's frame the blade runs along +x from the hinge and opens toward +y, so +y must be the pass direction
    n and the blade direction is t = rot(−90°) n; the hinge sits at the gap's end, centre − t · GAP / 2."""
    n = normal(k, to_b)
    t = np.array([n[1], -n[0]])
    hinge = gap_centre(k) - t * GAP / 2
    ang = np.arctan2(t[1], t[0])
    return np.array([hinge[0], hinge[1], 0.0]), np.array([np.cos(ang / 2), 0, 0, np.sin(ang / 2)])


def level_model(mx, geom_pos, body_pos, body_quat):
    return mx.tree_replace({"geom_pos": geom_pos, "body_pos": body_pos, "body_quat": body_quat})


def base_arrays():
    """The model's geom_pos with every filler at its gap (closed), and body_pos / body_quat (flaps parked)."""
    m, mx, d0, ids = model()
    gp = np.array(m.geom_pos)
    for k in range(NEDGE):
        c = gap_centre(k)
        gp[ids["fill"][k]] = [c[0], c[1], 0.0]
    return gp, np.array(m.body_pos), np.array(m.body_quat)


def physics(mx_level, d0, qpos, qvel, ctrl):
    d = d0.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

    def sub(i, d):
        return mjx.step(mx_level, d)
    d = jax.lax.fori_loop(0, SUB, sub, d)
    return d.qpos, d.qvel


__all__ = ["model", "EDGES", "EDGE_OF", "NEDGE", "gap_centre", "normal", "flap_pose", "level_model", "base_arrays",
           "physics", "R", "T", "GAP", "GM", "RAD", "GEAR", "DAMP", "MASS", "DT", "SUB", "NF", "FAR"]
