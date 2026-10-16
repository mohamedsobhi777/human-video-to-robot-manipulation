# Recorded experiment evidence

These are saved development/evaluation records, not a newly frozen benchmark. Numeric measurements are preserved; machine-specific path prefixes in JSON metadata have been rebased to `data/` for portability. Non-finite JSON literals, where present in old diagnostics, are serialized as `null`.

| Prefix / file | What it measures |
|---|---|
| `abl_{expert,egodex,mine}_{random,heldout,crowded}.json` | Same-policy demonstration-source ablation; 150 episodes per record |
| `film_session2_*.json` | Policy trained on my own clips; includes development runs and a layout-fitting diagnostic |
| `film_test_*.json` | Policy trained on EgoDex-derived keyframes |
| `act_*.json`, `diffusion_*.json`, `smolvla_styled_s30_*.json` | Development baselines trained on EgoDex-derived demonstrations |
| `replay_robots_*.json` | Embodiment-adapted controller replay, not learned-policy transfer |
| `session2_replay.json` | Earlier 240-step Panda replay protocol; do not merge with the 260-step embodiment results |
| `worldmodel_mine.json` | Full-episode predictions on 2,048 held-out-style simulated episodes |
| `worldmodel_mine_auc_audit.json` | Saved outcome scores/labels and a tie-aware AUC correction; original predictions unchanged |
| `wm_fidelity_mine.json` | Rolling-origin position-error analysis on 600 episodes / 9,600 start states |
| `bench_inference.json` | GPU model-call latency and amortized executed-action cost |
| `state_bc*.json`, `wm_policy_eval*.json`, `grasp_probe.json` | Learnability, world-model ranking and grasp-failure diagnostics |
| `session2_heldout_layouts.json`, `heldout_human_layouts.json`, `crowded_layouts.json` | Layout inputs for the corresponding evaluations |

`steps: 0` in an evaluation record means training was skipped because a checkpoint was loaded; it does not mean the policy was untrained. `wrong_fruit_in_bowl` counts any non-target fruit in the bowl at the end, even if the target was also placed there.

Some earlier diagnostics were overwritten during development. Do not infer an absent result from a slide or an old console summary; the final write-up uses the retained records. Training-seed variation, development selection and repeated scene layouts limit the conclusions. See [`docs/results.md`](../docs/results.md).

Write rerun outputs to `artifacts/`, not over these records.
