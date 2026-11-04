# From a first log to an autonomous Any% agent: bootstrapping architecture

> **Current implementation status:** The
> [V9 real Minecraft temporal target audit](V9_TEMPORAL_TARGET_AND_REAL_GOAL_AUDIT_2026_11_04.md)
> proves own-action/target-history observation, checkpoint+replay
> integration and a genuine self-achieved item-goal training test.
> It does NOT yet prove reliable inventory acquisition or autonomous
> skill composition. The milestones below remain research gates.


**Design decision / roadmap, November 3, 2026. NOT an implemented system or a performance claim.**
This is the long-horizon direction for `gmaincraft`. Read alongside
[OBSERVATION_POLICY.md](OBSERVATION_POLICY.md) and
[V7/V8 learning diagnosis](V7_V8_WOOD_LEARNING_FAILURE_DIAGNOSIS_2026_11_03.md).
If an experiment contradicts a hypothesis below, measure it, revise the
design, and version the change rather than training indefinitely on faith.

## 0. The question we must answer before scaling

A from-scratch agent that learns to find and collect wood does NOT thereby
know that making tools, exploring another dimension, fighting a boss, and
entering an exit portal are useful. Conversely, manually chaining
`get_wood -> make_pickaxe -> ... -> fight_dragon` would encode a human
speedrun and exclude interesting alternative strategies.

**Design target:** learn useful skills from actual experience, learn what
they *change* in the agent's known world, and learn to choose/compose/revise
those skills in pursuit of one externally specified terminal objective.
The trainer may expose local practice goals, but must not secretly dictate
the sequence of goals in an unrestricted run.

**One-sentence mental model:** the low-level policy learns *how*; an
experience-built outcome model learns *what happens if*; a high-level policy
learns *which possibility is worth pursuing now*. The agent can still
choose primitive actions directly.

Minecraft Any% success here means (a) defeat the real Ender Dragon and
(b) enter the resulting vanilla exit portal, on an unseen random survival
seed. The final goal is high success across independent unseen seeds and
eventually **under 30 minutes of in-game player time** (provisional
engineering definition: <=36,000 normal 20-TPS simulation ticks); also
record real elapsed wall time and startup/reset costs separately. Success
rate comes BEFORE speed: never let a fast loss outscore a completed run.
"Any new random seed" is an aspirational generalization target, not a
promise of 100% success under 30 minutes on literally every seed.
No direct full-game capability, hierarchy, crafting policy, or final
win evaluator exists yet.

## 1. What is actually present and what is missing

- **Present:** real Minecraft Java 1.21.3 with normal player control;
  bounded own status/inventory and experimental first-hit surface readings;
  first-log natural survival PPO with per-seed training, matched frozen
  evaluation, real inventory scoreboard, CUDA and resumable checkpoints.
  V7/V8 choose among 20 fixed *ordinary-input* move/look/attack/duration
  motor primitives; V8 had two real training wood pickups in ten fresh
  worlds, but is not reliable on held-out seeds or alternate rollout RNG.
  **New implemented baseline:** a player-only compressed self-play
  archive, checkpoint-safe recovery, a from-scratch goal-conditioned
  hindsight-imitation network and a frozen item-goal Minecraft skill
  evaluation path are in
  [DURABLE_EXPERIENCE_AND_SELF_GOALS.md](DURABLE_EXPERIENCE_AND_SELF_GOALS.md).
  These are infrastructure, not an established learned skill manager.
- **Historical separate skills:** aim/mine and pickup networks were trained
  in restricted environments and evaluated together after an **evaluator-
  scripted mined-event handoff**. This is not self-directed composition.
- **Missing:** a complete low-level vanilla input/UI action interface
  (use/place, hotbar, inventory, recipe-book/crafting UI, trade, combat);
  reliable player-visible dropped-entity sensing; full item/type inventory
  encoder rather than just log counts; task-conditioned skills; learned
  skill initiation/termination; off-policy reward-based outcome learning
  beyond the new positive-only hindsight imitation; long-lived episodic
  memory/belief; high-level goal selection; genuine unconstrained
  multi-skill learning; End victory detection and timed Any% scoring.
- **Risk identified NOW:** V5 observed only nine attack-on-wood decisions
  in 1,118 neural selections; V7/V8 improved the action search space,
  but neither learned a strong conditional response to seeing wood. More
  blind 24k-unit PPO is not the prerequisite for a speedrunner.

The old models are valuable measurements/checkpoints, **not** modules to
chain rigidly forever. Preserve them unchanged; introduce versioned new
interfaces and compare fairly.

## 2. Proposed architecture: one agent at several time scales

```text
        Final objective: dragon defeated + exit portal entered
                                  |
             High-level goal/option selector (learned)
               /              |             \
      explore/inspect    acquire/change     recover/adapt
               \              |             /
             Learned belief and outcome model
       (own history, observed resources, consequences,
         capability confidence, duration/risk estimates)
                                  |
               Goal-conditioned skill policy
        (shared encoder + adaptable skill repertoire)
                                  |
             Ordinary vanilla action interface
    (move, look, attack, use/place, inventory/UI, etc.)
                                  |
                    Real Minecraft world
                                  |
       Bounded player-knowable observation + memory
        -> experience/replay -> all learning levels

     Training-only reward/scorekeeper stays SEPARATE.
```

This is a **proposed functional decomposition**, not a demand for five
different GPUs/models or a fixed count of neural networks. Prefer shared
observation encoders, shared replay of the agent's own trajectories, and
goal-conditioned policies; specialized skill heads are allowed when
experimentally useful. The high-level decision-making may eventually be
implemented using option-critic, recurrent RL, learned planning, a compact
world model, or a hybrid. Choose by measured held-out transfer and compute.

### A. Primitive interaction: always retain a direct-control escape

Make ALL needed vanilla key/mouse/UI interactions possible through a
versioned action adapter; log each action's actual game-tick duration.
Do not provide "craft pickaxe," "locate fortress," "mine the targeted
block," or "trade for pearls" as privileged environment commands. A
learned skill CAN invoke many normal clicks/moves internally; it must
produce the same ordinary actions and obey the same reach, survival,
resource, and UI rules as the player.

The present V7/V8 20-choice motor is a useful scarcity experiment but not
necessarily the final expressive action space: it omits right-click,
hotbar selection and menus, and limits some combined actions. Prevent
this early motor discretization from constraining later novel strategies.
Allow general action duration/interrupts, especially for mining/continuous
interaction, without granting a one-click task success command.

### B. Skills as reusable *options*, not an ordered checklist

A learned option should expose a stable interface, conceptually:

- **Goal/intent:** an agent-grounded desired observable outcome, e.g.
  "increase inventory count of observed resource X," "change object Y,"
  "reach remembered place," or a learned latent state change. It is
  NOT a privileged target coordinate/recipe or a hidden-world ID.
- **Initiation estimate:** from current *observed/remembered* state, can
  this skill plausibly work? The estimate may be wrong and improve.
- **Internal policy:** takes ordinary player-knowable observations and
  chooses vanilla controls over multiple ticks.
- **Termination / interruption:** recognizes observed success, failure,
  timeout, risk, or changed circumstances. A missing tool, a dangerous
  encounter, or an item already acquired may change what to do next.
  **No evaluator-secret "block was mined" signal to switch skills.**
- **Learned outcome distribution:** estimates possible observed changes,
  success probability, game ticks, resource cost, and uncertainty.
- **Provenance:** observation/action/reward schema, training conditions,
  weight version, and held-out performance. No "mastered" label based only
  on training reward or a few lucky seeds.

The option may invoke other learned options in the future, but unroll to
ordinary actions and account for total duration (semi-Markov learning).
Multiple options may achieve the same goal; the selector must be free
to invent a new way or act primitively. Avoid a giant set of disjoint
single-item AIs with incomparable inputs and rewards. Share representation
and allow continual improvement/replay so wood experience also informs
stone, item use, and later materials.

### C. Learn causal, uncertain *consequences*, not a hardcoded recipe DAG

The agent records transitions:

```text
[what I knew] + [what I attempted] + [elapsed ticks]
        -> [what changed in what I now know]
```

Learn which options tend to produce useful resources/changes, which
prerequisites are *empirically* necessary, and which outcomes can substitute
for each other. Inventory deltas, actual observed block changes, crafting
UI feedback, own status, discovered locations and dimension changes are
natural labels. Maintain uncertainty and timestamps for stale observations;
the agent must revisit/verify facts when needed.

This lets an agent discover causal relationships among obtaining wood,
constructing objects, using tools, fighting, exploring and world events
without a hardcoded ordered "wood -> stone -> iron -> Nether -> End" plan.
A compact dynamics/outcome predictor and later imagined rollouts are
reasonable research paths, but should not be assumed accurate before
calibration on real Minecraft outcomes.

**Crafting knowledge boundary:** the agent may read the same in-game
recipe-book entries/UI and owned inventory that a skilled player can see,
or learn from actual interaction. Do not preload the complete recipe/
progression graph, hidden item drops, world-gen code, blaze/stronghold
locations, seed-derived structure directions, or an external walkthrough.
The curriculum may request "produce item X" as a *practice task*, but
cannot directly hand the policy the sequence or secret prerequisites.

### D. Learn *which goal/option to attempt* without a human route

The manager sees the same player-knowable observation and its own learned
memory/belief, plus predictions/skill metadata derived from **its own**
experience. Its available choices include attempts to use skills, explore,
gather information, repair failures, rest/recover, and primitive exploration.
Candidate goals can be proposed from achieved experience and visible/
remembered objects, inventory possibilities, novelty, and learned
outcome differences; do not require a complete predefined item list.

Train its choice policy with actual long-horizon returns and outcome
prediction, not hardcoded `if no wood then mine wood`. A planner may
search the **agent-learned** consequence model and revise plans when an
option fails; this is *not* a baked-in recipe DAG. Skill choice,
goal switching, and interruptions must work even when alternative paths
outperform our expectations. Have an ablation where the agent can ignore
the hierarchy and select ordinary actions.

## 3. Reward architecture: multiple training purposes, ONE final goal

**Most important rule:** do not assign fixed global rewards like +100 wood,
+500 stone, +2000 iron, +5000 fortress and expect their sum to discover a
speedrun. That both predefines a human route and invites farming the
wrong thing. Different rewards belong to different training levels:

| Layer | What qualifies for reward/feedback | What must NOT happen |
|---|---|---|
| **Terminal Any%** | Verified *real* dragon defeat and vanilla exit entry, with time as a secondary optimization **only among wins** | No reward for following an expected sequence or going toward a hidden stronghold |
| **Practiced goal/option** | Success/failure of a selected, *observable* state change (e.g. actual owned item increase), plus its elapsed game ticks and costs | No free inventory grants in scored play; no handoff triggered by evaluator-only world knowledge |
| **Outcome/model learning** | Predict the agent's next observation and achieved outcomes from its own actions; use prediction loss/real replay | Never import unseen map data or teacher-provided future facts as model input or labels |
| **Exploration/skill discovery** | Bounded novelty, information gain, achieved new capabilities, learning progress; contextual/task-conditioned | No endlessly repeatable bonus for seeing wood, running in circles, dying, or spam-crafting |
| **Safety/resource feedback** | Measurable survival, consumed items, failures, time wasted; compare with task success | No arbitrary anti-risk penalties that forbid unconventional valid routes |

A good default for a practiced goal is a sparse binary *verified outcome*
from a player-knowable inventory/UI/world change, with time/risk learned
as cost. For sparse data, use successful fragments replayed for other
*achieved* goals (hindsight relabeling) rather than fabricate scripted
expert demonstrations. Explicit failure episodes are equally informative.

The sparse terminal objective must stay **unmodified in final scoring**.
If shaping helps local training, scope it to a selected task, cap/dedupe
novelty and audit whether it changes what the policy chooses. Prefer
potential-difference shaping where assumptions fit:

```text
shaping = discount^(elapsed_game_ticks) * potential(next_belief)
          - potential(previous_belief)
```

This is a design tool, not magic: classic policy-invariance results apply
under their MDP assumptions and correctly handled discount/terminal
conditions. A learned partial-observation belief, approximated values,
changing goals, and termination/reset mistakes can break those guarantees.
Test with shaped vs sparse-reward frozen ablations. Never pass a teacher's
distance to an unseen End portal as a "potential" or feature.

Separate intrinsic and extrinsic reward **ledgers** and normalize across
tasks; avoid one enormous mixed total dominated by easy actions. High-level
selection should learn from option success probability, expected effect,
uncertainty, time and *terminal* return. Early training can use discovery
or competence rewards to get a foothold, then anneal/ablate them when
they conflict with winning.

### Practical anti-reward-hacking tests

- Let a policy repeatedly mine and replace the same item/area: verify it
  cannot accrue unlimited novelty/achievement credit.
- Let it collect wood forever: verify the Any% score does not improve
  unless it eventually wins (extra logs are telemetry, not final points).
- Swap a remembered visible target for a different visible target: does
  skill choice/action probability respond appropriately?
- Compare **true inventory** to mined-stat events, dropped-entity counts
  and repeated UI interactions; false successes are test failures.
- Ensure dying, resets, inventory drops, or world reloads cannot farm
  first-time rewards through mistaken epoch/counter resets.
- Evaluate a policy without shaped reward after freezing it, on unbiased
  unseen seeds and multiple independent action-sampling RNGs.

## 4. Curriculum without secretly writing the speedrun

Curriculum is a **training distribution and evaluation tool**, not the
deployment controller. Use training-only real Minecraft situations where
rare behaviors become learnable (e.g. naturally occurring forest-rich
starts, varied distances, existing objects). Label interventions clearly.
Never claim restricted success is unrestricted transfer.

Proposed stage gates; *none* beyond today's first-log experiments are
implemented:

| Stage | Capability and experiment | Graduation evidence before scaling |
|---|---|---|
| **0. Protect the interface NOW** | Version observation, action, goal/outcome, time and skill metadata. Keep raw self-collected transitions usable for later replay; preserve legacy models. | Frozen reproducibility, no observer leakage, no trainer/sensor mismatch |
| **1. Robust generalized interaction** | Unrestricted natural acquisition, mining continuity, visible dropped-item pickup, navigation/survival. Include varied material/object goals, not wood-only inputs. | Actual real inventory/object outcome on many unbiased held-out seeds **and** multiple rollout RNGs; gains over random, fixed and previous policies with uncertainty |
| **2. Inventory and UI mastery** | Learn hotbar, place/use, crafting/inventory interactions and recipe UI feedback in real Minecraft. Generalize across unseen item IDs, layouts, resource quantities. | Real task completion with no direct craft-command shortcut; items actually consumed/produced; new goals transfer |
| **3. Skill composition** | Learn option availability, success/failure, termination and switching, with shared memory/goal representation; practice mixed goals from naturally varying states. | *Without scripted phase changes*, autonomously reuse two or more skills in either order when needed; recover from blocked/failed skill; outperform fixed sequencing |
| **4. Learned planning/exploration** | Build uncertain causal outcome model, replay, hindsight goals, discovery and multi-step decisions across long episodes. | Unseen combinations / alternative paths, no recipe DAG or goal-order leakage; long-term objective progress outperforms flat policy and fixed-script controls |
| **5. Full Any% environment** | Add Nether/End interactions, combat, portals, dimension memory, proper exit scoring. Let the manager discover routes through experience. | First verified start-to-exit run on a held-out seed with normal inputs; then repeatability, success distributions and lower in-game time |
| **6. Speedrun optimization** | Optimize the **learned** successful strategy for expected success then low game time; retain alternative plans and continued discovery. | Increasing held-out win rate; eventually wins under 36,000 ticks on a meaningful independent seed sample; report both game-time and wall-time |

These stages are *evaluation milestones*, not a mandatory order of gameplay
or an assertion that "wood then stone then X" must be the policy's plan.
Tasks may be interleaved and explored out of order. Do not promote a
skill to "reliable" on 1/4, 2/10 or a single sampled rollout. Use enough
held-out worlds and independent policy RNGs to estimate confidence
intervals; pre-register seed streams, freeze model selection, and keep
test data separate from curriculum tuning. A numeric threshold should
be chosen from measured baseline variance and cost, not invented now.

### First small composition experiment (do this BEFORE hundreds of hours)

After **real, natural, reliable** wood acquisition and a legal item-use/UI
adapter exist, train a goal-conditioned controller to choose between two
learned tasks and primitive actions. Start with one meaningful real
Minecraft scenario in which doing task A alone or task B alone is
insufficient for the assigned *observable* outcome, and where their
order/necessity changes across scenes. The scenario may be designed for
TRAINING/diagnostics only, with no hidden target features given to the
agent. Score a *frozen* selector on unseen layouts/seed and alternate
action RNG. Compare:

1. A single flat policy with the same actions/observations/time budget.
2. The two options invoked in a fixed human-prescribed order (clearly
   labeled as an **engineering baseline**, not the final agent).
3. A learned manager choosing/interrupting options plus primitive fallback.

**Go/no-go:** learned management demonstrates independent switches,
recognizes failed/inapplicable skills from own state, and performs
meaningfully better than flat/fixed baselines **on new cases**. If it
doesn't, troubleshoot credit, state, and skill interfaces rather than
adding twenty more options.

## 4A. The missing bootstrap: how can the first dragon win ever occur?

The "learned consequences + high-level manager" abstraction alone does
NOT answer the toughest exploration problem. If the only whole-episode
reward arrives at the dragon/exit, then early random agents almost never
experience it; no correct credit-assignment algorithm can learn a useful
win-value estimate from **zero winning or near-winning experience**.
Likewise, learning wood, stone and crafting does not automatically reveal
the existence of dimension travel, a boss or an exit condition.

We should explicitly compare **two scientifically distinct bootstrap
regimes** instead of quietly smuggling human game knowledge into the
agent:

1. **Strict open-ended discovery (least prior knowledge):** train only
   from fresh ordinary survival starts, with player-observable exploration,
   item/interaction novelty, learned goals, in-game recipe/advancement UI
   if legitimately available, self-generated replay and experiments.
   Never introduce prepared late-stage scenes. It is purest under the
   user's constraints, but the search horizon may make End discovery
   prohibitively expensive on one RTX 4070 SUPER, even with model-based
   learning. Record that failure honestly.
2. **Explicit research-curriculum bootstrap (more feasible, optional):**
   *separately and transparently* expose the agent to real vanilla
   late-game situations in TRAINING using saved/constructed world states,
   WITHOUT scripted routes, item sequences, secret coordinates in policy
   observations, automated actions, or a manager deciding an order of
   skills. Examples could include genuine later-dimension interaction
   practice and boss-combat practice with normal controls and bounded
   sensory input. This gives knowledge that such states exist and is
   therefore **additional curriculum prior information** even though it
   does not teach a path to reach them. Do not silently choose this
   regime or claim that it is identical to strict unguided discovery.

Either way, the final scored agent must start in a truly fresh,
unmodified Overworld with normal empty inventory and discover/reach each
subsequent state through its own actions. A win from a late-game research
fixture does not count as a full-run win. A manager must learn to connect
experiences across contexts, not merely perform each separately.

**Critical experiment before attempting the full objective:** withhold
one consequence/goal ordering from a trained two- or three-skill manager
and test whether it can autonomously discover the ordering from its own
observable results. Next make the gap between skill contexts increasingly
long and uncertain. Compare total required real game interactions
against flat PPO, a fixed human-written sequence (diagnostic only), and
a model-free manager. If the option hierarchy cannot bridge even these
small gaps, it is not ready for End discovery.

## 5. Data efficiency and compute: do not waste the RTX 4070 SUPER

Today V7 and V8 each had only about 2,400 *neural action selections* in
6,000 six-tick game-time units, plus real JVM/worldgen costs. At that
density, very rare multi-stage discoveries may never appear in an
on-policy PPO batch. Increasing the tick count alone is not a long-
horizon training method.

Recommended research progression (not installed/implemented yet):

1. Preserve every *valid self-generated* observation/action/outcome
   transition with seed split, schema, action duration, option ID and
   exact player-known observations; no unfiltered world packets.
2. Measure the return on **replay/off-policy** methods and hindsight
   achieved-goal relabeling in small held-out tasks vs current on-policy
   PPO. Verify labels stay player-knowable and do not splice impossible
   transitions. Store raw datasets locally/ignored.
3. Try a compact recurrent/latent **own-experience** world/outcome model
   if it predicts long enough for skill choice; compare held-out
   calibration, action efficiency, and actual Minecraft results with
   a model-free selector. No external pretrained world model.
4. Introduce an option hierarchy only when flat goal-conditioning cannot
   manage duration and delayed credit at reasonable compute.
5. Keep a fixed *evaluation budget in Minecraft game ticks*, report
   environment steps, neural choices, PPO/replay updates, GPU/wall time
   and Minecraft restarts. Never equate simulated tick totals with
   independent neural experience.

DreamerV3 is **a research example**, not a drop-in target or a reason to
rewrite the project today. Its from-scratch Minecraft diamond result used
a significantly different action space, **abstract crafting and accelerated
block-breaking**, and substantially different GPU budget (**up to about
100 million environment steps and nine days on one GPU**); it did **not**
establish a from-scratch Ender Dragon speedrunner. Its benchmark also
used item-milestone rewards, which is not equivalent to our proposed
route-neutral final Any% objective. Likewise, purely
option-critic, HER or curiosity methods do not guarantee hierarchical
planning will emerge. Validate each component against cheap baselines.

## 6. Information boundary: knowledge versus cheating

**Permitted in BOTH train and scored play, subject to versioned visibility
and parity:** exact own coordinates/status/inventory, player-observable
recipe UI, observed block/entity IDs/properties, vanilla crosshair and
reach/break feedback, legitimately remembered places with timestamps,
past actions and the agent's own learned causal model/skill probabilities.
Direct memory reads are fine; screenshots are optional.

**Forbidden as policy input, skill arguments, learned-model labels or
high-level goal hints:** unseen chunk/structure/ore maps, hidden container
contents, world seed, hidden coordinates of End portals/strongholds,
privileged item locations, preloaded crafting/progression/route DAGs,
expert demonstrations, pretrained outside policies/world models, or
worldgen-derived route optimization. Do not encode these as scalar
"getting closer" rewards either.

**Trainer/evaluator-only:** hidden state to create balanced *training*
scenes, reset, check independent terminal success and report scores.
Keep this in separate code/data paths, enforce whitelists and parity
tests. Training curricula may know a private scene label to construct a
practice world, but the policy cannot be told it; final assessment uses
unmodified natural seeds and no intervention. Full game action fidelity
is required in scored play.

## 7. Evaluation and failure policy

- **Final score:** verified win + exit, then survival-conditioned in-game
  elapsed time and win-rate distributions. Report failed/time-out/death
  episodes; no "progress score" presented as a completed speedrun.
- **Data split:** independent training, validation and sealed held-out
  full natural seeds. Held-out seeds must not be selected by forest density,
  biome favorability, portal proximity or visible loot. Alternative policy
  RNGs and independent model inits test lucky samples.
- **Composition score:** task success, number of autonomous switches,
  recovery from failure, reuse of prior skills on changed goals, comparison
  to fixed order, random and flat baselines; no evaluator-controlled switch.
- **Anti-shortcut score:** detect policy access to hidden facts, trainer-only
  data leaking into replay/goal proposals, recipe/item tables inserted by
  humans, and task-specific reward farming. Sanity-check that changing
  irrelevant cosmetic presentation doesn't break a claimed general skill.
- **Regression score:** every new sensor, interface, task, reward family,
  option schema and action duration gets a version and a frozen comparison.
  Never silently reinterpret saved models/checkpoints or overwrite ignored
  worlds. Track independent per-task competence to detect catastrophic
  forgetting; allow old skills to be retrained, combined or discarded.

### Architecture decision gates, in priority order

1. **Before more big wood training:** fix target-conditional sustained
   mining sample efficiency and proper visible item pickup; prove transfer.
2. **Before dozens of resource-specific models:** define a reusable
   goal/observation/action/outcome interface and self-play replay store.
   Prefer shared generalization over hardcoded wood/stone/iron architectures.
3. **Before training an Any% manager:** prove a two-skill selector performs
   autonomous handoff, interruption, and recovery without privileged flags.
4. **Before adding an expensive learned world model:** show a compact model
   predicts useful *observed outcomes* and improves held-out choices
   above a simpler option-value baseline on this exact workstation.
5. **Before chasing 30 minutes:** prove actual wins, then optimize speed
   without sacrificing success or closing off alternative strategies.

## 8. Non-decisions: preserve the ability to discover better solutions

We are **not** committing to a particular number of subnetworks, a
pre-authored "tech tree," fixed human crafting recipes, one permanent
reward coefficient set, a single mandatory play order, or a world-model
algorithm before evidence. Goal semantics, option selection, novel
strategies, and alternative resource paths should remain learnable.
Curriculum scaffolding is temporary; a completed held-out run uses
the same bounded policy sensors and normal vanilla inputs, **without**
any teacher, phase switch, or special start.

## Research anchors (inspiration, not imported weights or demonstrations)

- Sutton, Precup & Singh (1999), *Between MDPs and semi-MDPs*:
  https://doi.org/10.1016/S0004-3702(99)00052-1
- Bacon, Harb & Precup (2017), *The Option-Critic Architecture*:
  https://ojs.aaai.org/index.php/AAAI/article/view/10916
- Ng, Harada & Russell (1999), *Policy invariance under reward transformations*:
  https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Andrychowicz et al. (2017), *Hindsight Experience Replay*:
  https://openai.com/index/hindsight-experience-replay/
- Hafner et al. (2025), *Mastering diverse control tasks through world models*
  (DreamerV3; Minecraft diamond is **not** Minecraft Any%):
  https://www.nature.com/articles/s41586-025-08744-2

**Implementation status:** design only. Existing V2-V8 experiments,
reward profiles, observed results, checkpoints and executable interfaces
remain unchanged by this document.
