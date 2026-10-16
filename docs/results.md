# Evaluation protocols and interpretation

## Task and success rule

`FruitToBowl-v1` has a bowl and three scaled YCB objects standing in for a persimmon, a pear and a mandarin. Success is checked from the physical state at the end of the rollout: the named target is inside the bowl, both other fruits are outside it, the gripper has released the target, and the bowl height has not changed by more than 3 cm. "Inside" means the fruit center is within 7.4 cm horizontally of the bowl center and less than 8 cm above it; there is no settling-velocity or bowl-orientation test. This is a custom task, not LIBERO or an official ManiSkill benchmark. See `FruitToBowlEnv.evaluate` for the exact geometric/state thresholds.

The learned policy observes two RGB views, robot TCP state and the parsed target identity. Demonstration generation and controller replay additionally use simulator object poses. Rendering a human-style replay from the phone's calibrated pose is a visualization; it does not make the learned policy a real-world visual policy.

## My recordings and splits

- 30 clips recorded by one person in one overhead-phone/table setup; three target fruits.
- Held-out clip identifiers: `IMG_5796`, `IMG_5798`, `IMG_5801`, `IMG_5809`, `IMG_5816`, `IMG_5819`.
- The other 24 clips supply the training styles. Human trajectories are not copied wholesale: keyframes are clamped and executed by a closed-loop controller in sampled layouts.
- 771 successful demonstrations per policy source; 30,000 FiLM optimization steps, training seed 0 for the source ablation.
- Each policy is evaluated on 150 random-layout episodes (seed 10000), 150 episodes cycling the six held-out layouts (seed 20000), and 150 crowded-layout episodes (seed 40000).
- The original target labels are manual. Camera calibration and appearance prototypes are estimated at session level, using the recorded session. Therefore the held-out result is about **policy behavior on reserved layouts/styles**, not an independent end-to-end perception test. The six layouts are not 150 independent recorded scenes.

The source ablation's table is reproduced in the repository README. An observed 96% versus 90% random-layout success is useful exploratory evidence, but one training seed and 150 episodes do not establish a robust general improvement. EgoDex supplies 38 styles versus 24 of mine, so its advantage cannot isolate data source from style diversity. The crowded differences are small. No zero wrong-fruit count should be read as a zero true error probability.

## Development baselines: public EgoDex, not my clips

All rows below use 771 **EgoDex-derived** demonstrations. They are distinct from the own-recording experiment.

| Policy, 30k training steps | Random-layout success | EgoDex held-out-layout success | Random-layout wrong fruit |
|---|---:|---:|---:|
| FiLM ResNet-18 | 97.3% | 98.0% | 0.0% |
| ACT | 13.3% | 14.7% | 16.0% |
| Diffusion Policy | 19.3% | 8.0% | 52.0% |
| SmolVLA | 22.0% | 20.0% | 6.7% |

FiLM receives a parsed target embedding and modulates its image encoders. ACT and Diffusion Policy receive the target as one-hot state; SmolVLA receives text. Pretraining, parameter count, input resizing, action execution chunks and optimization recipes differ. These are task-specific implementation baselines with limited tuning, not a controlled comparison proving one architecture generally superior. SmolVLA's own-data fine-tuning attempt stalled before any saved checkpoint; no score from that attempt is included.

The original development data are selected `pick_place_food` clips from EgoDex's published **test** split. This project used that public material for development/training; it does not report an official EgoDex benchmark score or claim an untouched public test set. The local within-project clip split is separate.

## Robot replay

The later embodiment protocol uses 260 control steps, per-robot command adapters, and human-derived styles from all 30 clips. Each robot succeeds on 30/30 recorded-style trials and a separate 128/128 random-layout set. The recorded-style set contains **16 fitted measured layouts plus 14 sampled replacements**: those 14 source clips started with one or more fruits already in the bowl. Those original starts are incompatible with this task's requirement that non-target fruits end outside the bowl.

The replacements are implemented explicitly in `scripts/replay_robots.py`; they must not be described as 30 exact-layout reconstructions. The earlier `session2_replay.json` used a 240-step Panda protocol and recorded 29/30 successes. It is not pooled with the later results. The three-arm result is **controller retargeting**, not a learned policy transferred without training. WidowX uses a different IK implementation, gripper convention and capped carry height.

## Inference timing

`bench_inference.json` reports median and 90th-percentile milliseconds per model call. At batch size one:

| Policy/configuration | Median call, ms | Actions executed per call | Amortized ms/executed action |
|---|---:|---:|---:|
| FiLM fp32 | 1.838 | 5 | 0.368 |
| FiLM fp16 + channels-last + compilation | 0.694 | 5 | 0.139 |
| ACT | 4.173 | 10 | 0.417 |
| SmolVLA base | 103.494 | 10 | 10.349 |
| Diffusion Policy | 864.521 | 8 | 108.065 |

Inputs are already prepared on the GPU. Timings exclude perception, camera transfer, image preparation, simulation, control and compilation warm-up. SmolVLA uses the public base model for this benchmark, with the same architecture as the fine-tuned model. Diffusion uses its configured 100 denoising steps, not an optimized sampler. The roughly 150× number is **per call**; amortized executed-action cost is roughly 75×. Neither number is an end-to-end robot speed-up. Closed-loop scores were measured using fp32, not revalidated for the compiled fp16 variant.

All timings, including FiLM, use the Python 3.11 / PyTorch 2.10.0 + CUDA 12.8 baseline environment. The own-data FiLM training/evaluation environment uses Python 3.10 / PyTorch 2.5.0 + CUDA 12.4. See the separate requirement files; do not merge the environments.

## World model

A five-member action-conditioned ensemble predicts low-dimensional simulator state, with an attachment mechanism for grasp/carry behavior. It does not generate camera images. Training uses 6,144 simulated episodes derived from my 24 training styles, including action drift and gripper flips. Testing uses 2,048 episodes derived from the six reserved styles.

- Full-episode outcome accuracy: **89.36%**; plan-following reference **83.94%**; majority baseline **52.93%**.
- Tie-aware outcome ROC AUC: **0.9541**. The original record reports 0.9650 from an implementation that assigned distinct ranks to tied ensemble scores. This was corrected during submission audit, without retraining or changing predictions; the original record is preserved and the scores/labels plus corrected calculation are in `results/worldmodel_mine_auc_audit.json`.
- Full-episode median final fruit-position error: **6.20 cm**; plan-following reference **8.00 cm**.
- Separately, a rolling-origin analysis over the first 600 test episodes / 9,600 start states evaluates horizons from 0.1 to 4.0 seconds. Median fruit error remains below 1 cm through **2.0 seconds**; at 1 second it is about **0.53 cm**.
- These rolling starts overlap and are not independent trials. The trust threshold is chosen for interpretation on this development analysis, not a held-out safety certificate. Median error conceals the tail.
- Ensemble spread correlates with short-horizon position error, but it is not a calibrated probability of task success. In the retained full-episode results the most-confident 80% actually have lower outcome accuracy than the least-confident 20%.

Earlier models developed on public data ranked closed-loop policies poorly (`wm_policy_eval*.json`). The own-data fidelity results do not establish that world-model planning or policy ranking works. The animation is open-loop prediction given initial true state and logged actions, not autonomous imagined control.

## Failed or unfinished work

- Raw trajectory replay is executable but difficult to imitate. Retained privileged-state cloning records show 25.8% success for raw paths and 76.2% for styled keyframes under their respective data-generation protocols. An earlier 91% styled run appears in the devlog but its full result record was overwritten; the final presentation uses the retained 76.2% result. This diagnostic is not a matched-data-size ablation.
- Five-finger Inspire hand: the scripted probe and replay diagnostics did not produce a successful fruit lift. This is a simulator/task-specific failure, not a general statement about dexterous hands.
- Phone teleoperation was operational but difficult for the person to use; no successful teleoperation training dataset resulted.
- PICO, SO-100 scene scaling, RL fine-tuning, real-robot transfer and world-model planning are not completed results.

The final submission contains working code and saved evidence, not every proposed experiment. Single-session data, simulator mesh approximations, development-set selection and limited training-seed replication remain the central limitations.
