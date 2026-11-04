# V7/V8: why unscripted natural-world PPO is learning slowly

**November 3, 2026; verified Minecraft Java 1.21.3, CUDA on RTX 4070 SUPER.**
These are bounded from-scratch research runs, **not** evidence of a trained
Any% speedrunner or reliable natural wood acquisition. Real world generation,
normal vanilla move/look/attack packets, and real Minecraft inventory/stats
were used. Model/world/console artifacts are local and Git-ignored.

## Diagnosis before new experiments

The previous training budgets were 6,000 or 24,000 **six-tick gameplay units**,
not millions of independent PPO actions. V5 in ten actual natural seeds made
**1,118 neural action choices**, only **21** of which started with vanilla
crosshair on wood. Nine choices also attempted attacking on wood, and
**zero** combined attack+unchanged camera while on wood. V6 produced 1,126
choices, 13 crosshair-wood, six attack-wood, one stationary attack-wood.
The older, longer V4 training had 3,721 sampled choices, 62 crosshair-wood,
25 attack-wood, nine stationary attack-wood. These rare successful action
prefixes explain the tiny amount of useful mining feedback.

The original four independently sampled categorical action heads choose
one of 6 movements, 7 camera changes, attack on/off, and 3 hold durations.
A random **stationary 72-tick attack** has theoretical probability
`1 / (6*7*2*3) = 1/252` before the independent chance the player is
already aiming at wood. This is a **search-space engineering problem**,
not proof that PyTorch/CUDA or the PPO gradient is broken. Actual status,
wood rewards, and performance/observations also need larger calibration.
The feed-forward model has only short own-action/visit history; it is NOT
a fully recurrent memory-planning speedrunner.

An offline controlled network-sensitivity probe using matching typed input
shapes found V6 almost indifferent to a wood-vs-nonwood crosshair:
`P(still attack)` roughly **0.0088** without wood vs **0.0087** with wood.
This is a diagnostic probe, **not** a held-out gameplay success metric.

## V7: one learned 20-way ordinary motor-action head

`wood_motor_v7` pairs ordinary vanilla actions with a duration (6,24,72)
in one categorical choice. `attack_stationary72` now receives about **1/20**
initially uniform probability, instead of **1/252**. Other actor-selected
choices cover movement, 24-pixel fine looking, larger looking, jumping,
attacking while moving/looking, and shorter stationary attacks. The
network **always** chooses the primitive itself. There is NO rule such as
"if wood then attack", target finder, auto-aim, scripted phase, hidden
world memory, or privileged teleport in play. V7 retains V5 reward
weights and V6's directly visible `crosshair_is_wood` and current-top24
surface wood fraction, but has a new strict checkpoint/action schema.
The V7 policy's coarsely encoded last-look history is a *direction bin*,
not a false claim that a small look has the same physical magnitude as
an old large look; own yaw/pitch still report actual orientation.

Actual **V7 training** `motorv7pilot_1103`: 10 never-repeated vanilla
noise train seeds, **6,000 game-time units / 36,000 Minecraft ticks**,
**2,427 neural actions, 82 CUDA updates**, 175 regular blocks mined,
503 newly observed wood surfaces, **one truly mined wood + one inventory
pickup** (episode9, units244/260), six deaths. About **2.51 ms MCio I/O
per game tick**, no performance regression. 57 choices started while
vanilla crosshair aimed at wood; only nine were stationary-attack actions.

## V8: separate reward ablation, same 20-way learned motor

`wood_sustain_v8` **does not change V7's observation or action contract**,
but is a DIFFERENT exact reward-profile checkpoint (new scratch weights,
no drift on resume). Reduce passive wood sighting reward from `.40` to
`.02` per new first-hit sampled wood, reduce incidental exploration and
ordinary block breaking bonuses. Add a new bounded **own vanilla crosshair
on wood + stationary attack command** reward of **0.75 per six-tick
sample, at most 12 samples/unique actual targeted wood block**, and keep
real visible crack progress plus substantial **actual wood mined and
inventory acquired** reward. The new per-block ledger tracks only live
visible target identity/XYZ, never scans unobserved world geometry, and
caps possible reward to limit repeat farming. This reward is a *proxy*
and can still be collected without a physical mined/picked log; never
confuse it with task completion.

Actual **V8 training** `sustainv8pilot_1103`: 10 fresh real survival
noise seeds, **6,000 game-time units / 36,000 ticks**, **2,423 sampled
neural actions and 82 CUDA updates**, 140 ordinary blocks mined,
730 new visible wood surfaces, **two real mined-and-inventory wood**
(episode2 units120/122; episode3 units260/268), eight deaths. About
**2.53ms game I/O/tick**, 21.0 total bounded sustained-attack reward.
This was **2 wood pickups vs V7's 1** over ten seeds, but no reliable
learned strategy is proven, and reward magnitudes differ by design.

## Held-out frozen evaluation, matched Minecraft seeds and time

Each model plays a heldout seed for 600 six-tick units (3,600 game ticks).
Train and test seed streams are disjoint, no optimizer updates in eval;
wood acquisition means real own inventory gain, not surface recognition.

| Frozen controller | Heldout indices | Actual wood mined | Real wood inventory gains | Ordinary mined | Deaths |
|---|---|---:|---:|---:|---:|
| V7 sampled (82 updates) | 6-7 | 0 | 0 | 19 | 3 |
| V7 sampled (82 updates) | 8-9 | 0 | 0 | 22 | 2 |
| V8 sampled (82 updates; default action RNG) | 6-7 | 2 | **1** | 22 | 1 |
| V8 sampled (82 updates; default action RNG) | 8-9 | 0 | 0 | 26 | 1 |
| Uniform random *V8 motor* | 6-7 | 2 | 0 | 51 | 1 |
| V8 **alternative action RNG=813777** | 6-7 | 0 | 0 | 11 | 4 |

**Inference caveat:** V8's single default-sample held-out pickup did NOT
replicate with another valid action RNG on the same seeds; random also
mined wood but failed to pick up drops. V8 is **not validated as better
than random across enough unseen worlds or alternate rollout RNGs**.

A frozen network sensitivity probe, changing only the typed visible
crosshair wood ID/flag while keeping a synthetic plausible player state
fixed, found:

- V7 `P(stationary attack)` **10.90% vs 10.97%** with wood; total variation
  in motor probability distribution about **0.012**.
- V8 `P(stationary attack)` **6.21% vs 6.48%** with wood; total variation
  about **0.010**.

These inputs are controlled counterfactuals, not full naturally sampled
observation distributions. They indicate neither network has learned a
strong **conditional** "wood targeted -> hold attack" association,
consistent with the low counts. V8 can gain while still not using the
correct conditional rule.

## Stability and reproducibility

One initial V8 frozen evaluation attempt on heldout seed8 crashed during
Minecraft client startup with NVIDIA 595.104.02
`libnvidia-glcore.so` **SIGFPE**; NOT a model-learning failure. The
incomplete audit was preserved (`fresh_eval_v8_frozen_89.jsonl` and
associated console/JVM hs_err). Repeated exact seeds in a *new* named
frozen evaluation `v8_frozen89_retry1`, which completed. Host driver,
network, world generator, original CUDA setup and old checkpoints were
left unchanged. All shell programs completed with verified full journal
and model checkpoint; Gamescope parent may show a known post-child 139.

Locally retained raw evidence:

- `logs/fresh_stream_motorv7pilot_1103.jsonl`
- `logs/fresh_stream_sustainv8pilot_1103.jsonl`
- `checkpoints/fresh_stream_motorv7pilot_1103.pt`
- `checkpoints/fresh_stream_sustainv8pilot_1103.pt`
- `logs/fresh_eval_v7frozen_67.jsonl` and `logs/fresh_eval_v7frozen_89.jsonl`
- `logs/fresh_eval_v8_frozen_67.jsonl`, `logs/fresh_eval_v8_frozen89_retry1.jsonl`
- `logs/fresh_eval_v8_random67.jsonl`, `logs/fresh_eval_v8_alt_rng67.jsonl`
- `logs/v7_v8_final_pure_1103.txt` — **149 pure tests pass, none skipped**.

## Next steps: change learning methodology, not just PPO reward

The agent is not "failing to learn anything": it learned some general
mining and locomotion behavior; it hasn't formed the crucial target-aware
conditional mining skill. Do NOT simply resume 24k/forever on V7/V8.
A sensible next experiment is **denser, realistic exposure to naturally
occurring wood** while keeping normal vanilla gameplay and unknown heldout
seeds. Investigate a *train-only* curriculum that chooses truly
generated forest-rich worlds (never a placed/pre-aimed tree or fake
block), separately from an unbiased full64 heldout population; document
selection criteria and any bias. Every candidate must be measured on real
Minecraft scenes. Another option is trained policy state with bounded
history of the player's **already observed** target/block and visible
break progress, so it can distinguish "continue this attempted mine"
from "random fresh action". These are not implemented yet.

Any new curriculum requires a fresh, versioned training setting: preserve
all existing raw worlds/checkpoints, do not leak seed/terrain filters to the
player policy, and use **multiple independent rollout RNGs** and sufficient
held-out seeds for true inventory pickup comparisons. An early acceptance
criterion should measure whether a model actually changes its stationary
attack behavior when a known wood crosshair and active break stage appear,
not just whether it earns shaping rewards.
