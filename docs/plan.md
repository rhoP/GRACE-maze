# GRACE-maze: the hierarchy on a MuJoCo point-mass maze

Started 2026-11-10 as a two-day spin-off of GRACE-bomber (`../GRACE-bomber` `docs/plan.md` §9, which passes every
criterion). The user asked for a more mainstream application than our own grid games. Point-mass maze navigation
in MuJoCo is a standard hierarchical and goal-conditioned RL testbed; Ant is stage 2 if this works.

What is new compared with the bomber:
- **Continuous control:** force actions on a damped point mass (top speed 2, braking distance up to 0.6) with
  contacts simulated by MJX.
- **Physical commitments:** one-way hinged flaps and doors that spend a key.
- **A momentum hazard:** pits kill on entry.
- **A conservative analytic in-game check** (braking distance) instead of the bomber's exact `viable`.

The abstract layer, faces, Φmax, shield, masks and regret identity stay exact. P1 becomes statistical.

Nothing here imports GRACE, the platformer or the bomber (`tests/test_core.py::test_standalone`).

## 1. Simulation and game (`maze/sim.py`, `maze/game.py`)

- **Frame:** 4 × 4 rooms of 4 × 4 units, walls 0.2 thick. Each edge between rooms has two static wall segments
  around a 1.2-wide gap.
- **The gap holds a filler box** (a wall, a closed door or a latched flap) or nothing.
- **Agent:** a sphere of radius 0.2 on two slide joints, with damping 1 and force ±2 per axis (top speed 2). The
  timestep is 0.02 and the control step 0.1 s (5 substeps).
- **One compiled `mjx.Model` serves every level.** Per-level geometry goes into the batched model fields
  `geom_pos` (fillers in place or parked) and `body_pos` / `body_quat` (flap hinges).
  - Mocap bodies are not used: MJX does not collide the jointed child of a mocap body (checked; plain MuJoCo does).
- **Flaps:** a hinged blade with range [0, 100°] opens when pushed from its *from* side and rests on its limit from
  the other.
  - An open flap also let the agent turn round and come back (15 of 30 turnaround rollouts leaked).
  - So the game **latches** the flap the moment the agent's centre crosses into the *to* room: its filler closes
    the gap. Re-entering the from room by another route still works.
  - The filler appears around an agent whose centre is just past the centreline, and contact pushes it on into the
    *to* room (no leak in any test).
- **Actions:** (fx, fy, use) in [−1, 1]³. `use` > 0.5 within 0.7 of a closed door, with a key held, spends the key
  and opens the door.
- **Events:**
  - the centre in a pit is death;
  - within 0.35 of an item is a pickup: a key adds a key (at most 3), a goal ends the episode with its payoff;
  - the room comes from the centre's position.
- **Throughput:** the solver is capped (`iterations=4`, `ls_iterations=8`) and contacts are frictionless
  (`condim=1`). MJX runs a batch until every env's solver converges, so the defaults gave 1.7k steps/s on bank
  states. The skill env now runs at 16.5k env steps/s on the laptop CPU, and PPO at 6.6k with 512 envs.

## 2. Levels (`maze/levels.py`)

- **Layout:** GH × GW rooms (3 × 3 for `mtrain`, `mtest` and `mpref`; 4 × 4 for `mbig`), joined by a spanning tree
  plus 1–2 extra links.
- **Link kinds:**
  - `door`: 35%, at most 4;
  - `flap`: 25%, at most 4, random direction;
  - `open`: the rest.
- **Pits:** 0–2 per room, 0.5–1.1 on a side, kept 1.0 from every gap and 0.7 from the room's centre.
- **Items:** goals A / B / C in three rooms other than the spawn room, 0–2 keys, and 1 key at the start.
- **Ranking:** goals are relabelled by cost (key uses and key pickups before them), and payoffs (1 / 0.6 / 0.3) go by
  rank. `mpref` uses the rankings with A last.
- **A level is accepted when:**
  - ≥ 2 goals are reachable;
  - some transition shrinks the face;
  - A is strictly the cheapest;
  - in every room, the gaps, items and spawn are connected on the planner grid.

## 3. Abstract layer and gates

- **State:** (room, keys, doors opened, flaps latched, items gone, end).
- **Options:**
  - `go` through an open link, an opened door, or an unlatched flap from its from room (it latches);
  - `open` a closed adjacent door (−1 key);
  - `get` an item.
- **Exactness:** every abstract transition the game can make is one of these options, so, unlike in the bomber,
  there are no off-plan transitions.

**P2 (abstract model, 50 levels per split; `runs/gates/p2.json`): passes.** C has 0 regret and 0 mistakes, D makes
0 mistakes, and the identity holds on every episode. U's mistake rate is 0.42–0.84, and C needs ≤ 9 options at p95.

**The oracle** (`maze/oracle.py`) plans on a 0.1 grid inside the current room, extended through the option's gap,
around pits and every other item. A PD controller follows the plan with a speed cap of 1.2 and presses `use` within
0.6 of the door.
- **Through the game** (5 levels per split): C has 0 regret, 0 deaths and the identity on every episode, taking
  40–140 ticks.
- **Failures:** 2 failed executions in about 500 options, from momentum carried over between options.

**P1 (statistical; `scripts/gates.py --p1`; laptop, 4 levels per split; `runs/gates/p1_*.json`): passes.** All 354
options reach exactly `abstract.succ` (0 plan failures, stalls or wrong outcomes). There are 0 leaks in 21,504 flap
and 20,352 door rollouts, and 0 door-use errors. On each level,
1. every option from up to 12 reachable abstract states, starting at rest at a random point, reaches exactly
   `abstract.succ`;
2. flaps are one-way: from the to side (unlatched, and latched), 64 rollouts each of random actions, PD aimed
   through the gap and a full-force ram never reach the from room;
3. doors need a key: without one, 30-tick rollouts with `use` pressed never cross or open the door; with one, one
   use opens it and spends exactly one key.

Rollouts start with every item gone. The first version counted 19 door "leaks" on `mpref`, all of them a key picked
up mid-rollout or a detour through other rooms in 60 ticks.

The pass bar is 0 leaks and 0 door errors. A small rate of oracle stalls is reported, not gated, because the
shielded episode retries a stalled option.

## 4. Arms and criteria (fixed before any run)

| Arm | Definition |
|---|---|
| C / D / U | the exact manager; uniform with the option shield; uniform without it (abstract model) |
| D+nr / D+lv | D with the no-revisit / count-based mask: the reference a learned manager must beat |
| E0 / E (+nr, +lv) | the learned manager without / with the shield (and the masks) |
| H+lv | E+lv's manager with the learned skill in the game, without the shield or the check |
| **G+lv** | E+lv with the learned skill and the in-game check (`hier.checked_step`): the headline arm |
| G+nr / G | the same with `no_revisit` / with no mask |
| A / A:shield | flat PPO on the same window plus the payoff, the items' offsets and the abstract flags; A:shield adds the check at test |

- **The in-game check:** each tick it takes the first allowed candidate among: the policy's action, the same force
  rotated by ±45° and ±90°, then the brake. A candidate is allowed when its successor is alive, does not lower
  Φmax, and is safe. Two changes came out of the smoke run, before any pilot:
  - the safety test is waived when the current state already fails it (braking cannot leave such a state);
  - the rotated candidates let the agent slide round a zone the policy pushes into.

  With pure braking and square item zones, G+lv braked on 58% of `mpref` ticks and timed out.
  - *Safe:* the box swept by the move plus the full braking distance per axis
    (m/c |v| − F m/c² ln(1 + c|v|/F), plus a margin of 0.05) meets no forbidden zone.
  - *Forbidden zones:* the pits, plus the trigger zones of Φmax-lowering events from the current abstract state:
    crossing an unlatched flap (a rectangle past the gap's centreline) and picking up an item (its pickup disc, radius
    0.35 + 0.02).
  - The braking bound holds against MuJoCo rollouts (`test_braking_bound`).
- **Skills and flat A** use the policy's mean action at test.
- **Manager:** `nraa` (bomber §7) is trained with `no_revisit`; `least_visited` is used at test.

**Criteria:**
1. G+lv's `mpref` regret ≤ 10% of uniform (U's regret), and also on `mbig`.
2. A's regret ≥ 2× G+lv's, or A's mistake rate > 0.1.
3. C, D and E make 0 mistakes, and the identity holds on every episode.
4. Reported:
   - δ by option type;
   - deaths (pits) in G against H;
   - the share of ticks on which the check braked;
   - E+lv against D+lv, and the agreement with C.
5. Protocol: seeds 0–1 (the pilot) set the budgets; the final run uses fresh seeds 2–4 with the budgets written here
   first.

## 5. Laptop smoke run (`runs/smoke/seed0`, 2026-11-10; every arm end to end)

- **Setup:**
  - bank: 30 levels, 1,440 entries (2 oracle stalls);
  - skill: 3M steps, reaching a return of 0.92;
  - manager: 20k decisions;
  - flat: 2M steps;
  - evaluation: 6 levels per split.
- **The identity** holds on every episode of every arm.
- **G+lv:** 0 mistakes and 0 deaths; it intervenes on ≤ 0.7% of ticks. Its regret comes only from timeouts of the
  undertrained skill (δ 0.11–0.15).
- **H+lv** (no shield or check): mistakes on 50–100% of episodes, and one pit death on `mbig`.
- **Flat A:** reaches no goal at 2M steps.

At this size the numbers say nothing yet about the criteria.

Figures: `plots/levels.png` (one level per split with its room graph: doors dashed, flaps as arrows) and
`plots/episode_oracle_mpref0.gif`.

## 6. Cluster (`scripts/cluster/submit.sh`, all CPU)

1. `p1`: 25 levels per split, one split per task.
2. `bank`: 1,000 `mtrain` levels in 20 shards.
3. `pilot`: seeds 0–1, with skill 30M steps, flat 200M and manager 1M decisions (`nraa`), 50 eval levels.
4. `mgr`: the 4 manager variants × seeds 0–1, abstract arms only.
5. `final`: seeds 2–4, with the budgets from the pilot recorded here first.

## 7. Pilot (seeds 0–1, 2026-11-10/11; `runs/slurm/maze-*`): hier arms done, flat still training

- **P1 on the cluster (25 levels per split):** passes. 2,332 options are all correct (0 plan failures, stalls or
  wrong outcomes); 0 leaks in 114,816 flap and 132,480 door rollouts; 0 door-use errors.
- **Bank:** 70,950 entries from 1,000 `mtrain` levels (go 19,457 / open 1,724 / get 2,469 options; mean 16–23
  ticks); 256 failed oracle executions (about 1%, momentum carried over between options; not in the bank).
- **Skill:** 30M steps, return 0.926 (flat after about 10M); 3.5k steps/s on 8 CPUs.
- **Manager (`nraa`, 1M decisions)** and the `mgr` variants: E+lv is 0% of uniform on every split, seed and variant,
  as is D+lv. E+nr is 0–20% on `mpref` and 19–63% on `mbig`. Task `nr` seed 0 printed nothing after its arguments.

| % of uniform (mistake rate) | seed 0 `mtest` | `mpref` | `mbig` | seed 1 `mtest` | `mpref` | `mbig` |
|---|---|---|---|---|---|---|
| **G+lv** | 0.0 (0) | **0.0** (0) | **4.2** (0.04) | 0.0 (0) | **8.7** (0.04) | **5.3** (0) |
| G+nr | 85 (0) | 45 (0) | 61 (0.02) | 18 (0) | 11 (0.04) | 42 (0) |
| G | 229 (0) | 159 (0) | 211 (0.02) | 61 (0) | 53 (0.02) | 106 (0) |
| H+lv | 102 (0.64) | 93 (0.80) | 90 (0.58) | 41 (0.28) | 51 (0.50) | 42 (0.28) |

- **Criterion 1 passes** on both seeds (G+lv ≤ 8.7% on `mpref`, ≤ 5.3% on `mbig`). Criterion 2 waits for flat A:
  its return is still 0.0 at 60M steps.
- **Reported:** δ is 0.6–3.2% per option. Deaths: G 0, H 1–2% of episodes on seed 0. The check intervened on
  0.4–2.1% of ticks, except **17% on seed 1 `mpref`**.
- **G made a few in-game mistakes** (2 of 50 episodes on seed 0 `mbig` and on seed 1 `mpref`). Traced with the
  saved manager and skill (`scripts/check_diag.py`, `scripts/check_trace.py`; the laptop reproduces the cluster's
  numbers exactly):
  1. *`mpref` 26:* with no candidate both safe and Φmax-keeping, the check braked, and braking lowered Φmax.
  2. *`mpref` 35:* a physics artefact. The latched flap's filler appears around the agent, and contact pushed it on
     at 3.0 units/s (the top speed is 2). From there no action avoids the cheap goal 1.7 past the gap.
  3. *`mpref` 15 (timeout):* the deadlock behind seed 1's 17% braked share. The skill passes the cheap goal at 0.36
     (pickup 0.35, zone 0.37) on its way to a door, and the rotated candidates alternate toward and away from it.
- **Changes before the final (2026-11-11):**
  - `game.step` places a latching agent clear of the filler (along the pass direction, velocity unchanged).
    `test_flap_one_way_and_latches` now bounds the speed; without the fix it measures 2.44. P1 (laptop, 4 levels
    per split) still passes with 0 leaks.
  - `checked_step` falls back to a Φmax-keeping candidate before the brake, and its candidates add the policy's
    force slid along the nearest active zone (and slid with a push away from it).
- **G+lv re-evaluated with the changes** (saved pilot manager and skill; % of uniform, mistake rate, braked share):

  | | `mtest` | `mpref` | `mbig` |
  |---|---|---|---|
  | seed 0 | 0.0 (0), 2.9% | **0.0** (0), 1.6% | **0.0** (0), 0.9% |
  | seed 1 | 0.0 (0), 0.9% | **3.6** (0), 18% | **5.3** (0), 0.4% |

  0 mistakes on every split and seed. Seed 1's two losses are timeouts: the `mpref` 15 deadlock (still there, and
  nearly all of the 18% braked share), and on `mbig` 22 a `go` that the skill never completes (δ, no braking). The
  zone margin is kept. Loosening it to let the skill pass at 0.36 would trade a timeout for unchecked integration
  error.

## 8. Final (seeds 2–4): budgets fixed here before submitting

- The game changed (latch placement), so P1 and the bank are re-run first (`p1`, then `bank`), then `final` after
  the bank with: skill 30M steps (its return is flat from about 10M), manager `nraa` 1M decisions, flat 200M steps,
  50 evaluation levels per split.
- Criteria as in §4. The flat pilot (old dynamics) finishes separately and is reported for criterion 2's budget only.

**Pilot flat A (seed 0, old dynamics, 200M steps):** the training return stayed 0.0 throughout; at test it reaches no
goal (timeouts on every episode), so A is 295 / 170 / 251% of uniform on `mtest` / `mpref` / `mbig`.

## 9. Final (seeds 2–4, budgets as in §8; `runs/final`, job 25056733, 2026-11-11): hier arms done, flat training

- **P1 on the cluster (new dynamics, 25 levels per split):** passes. 2,332 options are all correct; 0 leaks in 114,816
  flap and 132,480 door rollouts; 0 door-use errors.
- **Bank:** 71,634 entries (go 19,626 / open 1,755 / get 2,497).
- The abstract arms are identical to the pilot's (same evaluation levels): C, D+lv and E+lv are 0% everywhere.

| % of uniform (mistake rate), braked share | `mtest` | `mpref` | `mbig` |
|---|---|---|---|
| **G+lv** seed 2 | 6.1 (0), 21% | **6.1 (0.02)**, 1.1% | **0.0** (0), 1.2% |
| **G+lv** seed 3 | 0.0 (0), 7.0% | **0.0** (0), 1.2% | **0.0** (0), 1.3% |
| **G+lv** seed 4 | 0.0 (0), 3.2% | **5.8** (0), 1.1% | **15.8** (0), 1.4% |
| G+nr (seeds 2 / 3 / 4) | 30 / 10 / 12 | 18 / 4 / 24 | 51 / 32 / 53 |
| G | 121 / 58 / 67 | 121 / 72 / 97 | 146 / 104 / 168 |
| H+lv | 55 / 50 / 39 | 73 / 53 / 75 | 46 / 45 / 74 |

- **Criterion 1:** `mpref` passes on every seed (≤ 6.1%). `mbig` passes on seeds 2 and 3 (0%) but **misses on seed 4
  (15.8%: 3 of 50 timeouts)**; the mean over seeds is 5.3%.
- **One in-game mistake under the check** (seed 2 `mpref`, 1 of 50 episodes), and seed 2 `mtest` braked on 21% of
  ticks (a deadlock, as in the pilot's `mpref` 15). Both to be traced with the saved models.
- Deaths: G+lv 0; H+lv 1 option death on `mbig` for seeds 2 and 3.
- Flat A at about 80M of 200M steps: training return 0.0 (seeds 2, 4) and 0.09 (seed 3).
- **Traced with the saved models** (`scripts/check_diag.py`, `scripts/check_trace.py`; the laptop reproduces the
  cluster's G+lv results):
  1. *The mistake (seed 2, `mpref` 23) is a physics bug.* After the agent latches a flap, the blade (a free hinged
     box, still open at about 57°) springs shut through the to room and sweeps the agent along. 3 ticks after the
     latch it pushed the agent 0.013 per tick toward the cheap goal, 0.85 past the wall. The braking model leaves
     this push out, so a safe state had no safe successor, and even full braking (speed 0.05) was swept into the goal.
     Two weaknesses in the check showed up on the way: the pilot's fallback took the policy's force before the brake,
     and once a state failed the safety test the test was waived, so the policy could push on.
  2. *The seed 2 `mtest` braking (21%)* is one deadlock (level 32, an `open` option that braked on about 80% of its
     ticks). It is gone after the changes below.
  3. *The seed 4 timeouts are the skill (δ).* On `mbig` 7 the agent circles beside the spawn room's flap gap for
     300 ticks per attempt and never pushes through; the check does not intervene. Seeds 2 and 3 solve the same
     levels.
- **Changes (2026-11-11), made after seeing the final's hierarchy results:**
  - `game.step` holds a latched flap's blade shut (its angle and velocity set to 0; it lies inside the filler).
    `test_flap_one_way_and_latches` asserts this.
  - `hier.slack`: the safety test as a signed clearance (`safe` is slack ≥ 0, unchanged). `checked_step` no longer
    waives safety in an unsafe state; it requires the successor to be no less safe. Its fallback order is now
    safe and Φmax-keeping, then the brake if it keeps Φmax, then any Φmax-keeping candidate, then the brake.
- **G+lv re-scored with the changes** (the final's saved models, laptop; % of uniform, braked share):

  | | `mtest` | `mpref` | `mbig` |
  |---|---|---|---|
  | seed 2 | 0.0, 1.2% | **0.0**, 1.3% | **0.0**, 1.2% |
  | seed 3 | 0.0, 7.7% | **0.0**, 1.2% | **0.0**, 1.3% |
  | seed 4 | 0.0, 3.4% | **7.9**, 0.9% | **15.8**, 1.3% |

  0 mistakes on every split and seed. Seed 4's losses are all skill timeouts (3 of 50 on `mpref` and on `mbig`).
- **Status:** this table is diagnostic only. Both changes were chosen after looking at seeds 2–4, so they are tested
  on fresh seeds in §10.

## 10. Second final (seeds 5–7, `runs/final2`): the changed game and check, budgets as in §8

- The game changed again (the blade), so P1 and the bank are re-run first, then `OUT=runs/final2 SEED_OFFSET=5
  bash scripts/cluster/submit.sh final`. Budgets, arms and criteria are unchanged (§4, §8).
- Criterion 1 is judged per seed, as for the pilot. The weak skill seed (seed 4) is a known risk; nothing in the skill
  changed.
