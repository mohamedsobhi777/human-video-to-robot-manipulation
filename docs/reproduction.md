# Reproducing the submission

Run all commands from the repository root. The default workflow needs Linux, an NVIDIA GPU with working Vulkan rendering, FFmpeg, and Python 3.10. The original GPU has 16 GB VRAM. Baselines use a **separate Python 3.11 environment**; do not install their requirements over the core environment.

## 1. Core environment

```bash
python3.10 -m venv .venv-fruit
source .venv-fruit/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m mani_skill.utils.download_asset ycb -y
python scripts/check_env.py artifacts/smoke
python -m unittest discover -s tests -v
```

The smoke test renders the task, verifies that reset layouts are unsolved, drops each target into its bowl and asserts three successes. It does not test manipulation skill. For the optional simulator throughput probe, add `--throughput-envs 512`. On machines with a ROS installation, unset an inherited ROS `PYTHONPATH` if it interferes with this environment. FFmpeg/FFprobe must be available on `PATH` for video processing.

## 2. Download the models, human plans and original recordings

The [submission-v1 release](https://github.com/mohamedsobhi777/human-video-to-robot-manipulation/releases/tag/submission-v1) contains:

| Asset | Contents |
|---|---|
| `film_session2.pt` | FiLM policy trained on my 24 training clips' multiplied demonstrations; normalization buffers included |
| `worldmodel_session2.pt` | World-model ensemble trained on the same training styles with perturbations |
| `session2-plans.pkl` | All 30 processed hand/object plans, with clip identifiers; no raw frames or audio |
| `session2-recordings.zip` | All 30 original `.MOV` recordings, including original audio, plus the manual `targets.json` labels (205 MB) |

```bash
mkdir -p data/downloads data/fruitbowl/wm_mine data/session2/processed
gh release download submission-v1 \
  --repo mohamedsobhi777/human-video-to-robot-manipulation \
  --dir data/downloads
(cd data/downloads && sha256sum -c ../../docs/checkpoints.sha256)
cp data/downloads/film_session2.pt data/fruitbowl/
cp data/downloads/worldmodel_session2.pt data/fruitbowl/wm_mine/model.pt
cp data/downloads/session2-plans.pkl data/session2/processed/plans.pkl
cp results/session2_heldout_layouts.json data/session2/processed/heldout_layouts.json
```

Alternatively download those assets in a browser. Only unpickle plans from a trusted source: Python pickle can execute code. Model loading uses PyTorch's `weights_only=True`. SHA-256 values are in [`checkpoints.sha256`](checkpoints.sha256).

To inspect or reprocess the original recordings, extract the archive into the session directory:

```bash
unzip data/downloads/session2-recordings.zip -d data/session2
```

This creates `data/session2/raw/IMG_5791.MOV` through `IMG_5820.MOV` and `data/session2/targets.json`. The files are byte-for-byte originals; the 24-training/six-held-out split remains unchanged.

## 3. Evaluate the own-data policy

Training demonstrations are **not required** to load and evaluate the released policy. First run three episodes to check the environment, then the saved protocols:

```bash
python scripts/image_bc.py --film --load data/fruitbowl/film_session2.pt \
  --episodes 3 --seed 10000 --out artifacts/policy-smoke.json

python scripts/image_bc.py --film --load data/fruitbowl/film_session2.pt \
  --episodes 150 --seed 10000 --out artifacts/mine-random.json \
  --video artifacts/mine-random.mp4

python scripts/image_bc.py --film --load data/fruitbowl/film_session2.pt \
  --episodes 150 --seed 20000 --layouts-file results/session2_heldout_layouts.json \
  --out artifacts/mine-heldout.json

python scripts/image_bc.py --film --load data/fruitbowl/film_session2.pt \
  --episodes 150 --seed 40000 --layouts-file results/crowded_layouts.json \
  --out artifacts/mine-crowded.json
```

These reproduce the intended evaluation recipe, not a promise of bit-exact physics across GPU/driver versions. Each evaluation executes 200 policy actions, each held for two simulator control steps. FiLM predicts chunks of 10 and executes 5 before replanning. Success is measured at the end, not at any transient successful frame.

## 4. Regenerate the own-data demonstrations and train FiLM

This path starts from the released processed plans, so neither WiLoR nor raw recordings are needed. The six held-out clips are excluded explicitly.

```bash
python scripts/make_expert_dataset.py \
  --plans data/session2/processed/plans.pkl \
  --exclude results/session2_heldout_layouts.json \
  --episodes 771 --out data/fruitbowl/demos_session2

python scripts/image_bc.py --film --demos data/fruitbowl/demos_session2 \
  --steps 30000 --train-seed 0 --save data/fruitbowl/film-retrained.pt \
  --out artifacts/film-retrained-validation.json
```

The generator should print `24 training plans`. It samples new layouts and saves successful episodes only, as state/action NPZ files plus over-shoulder and wrist-camera MP4s. A style is not a raw frame-by-frame hand replay; the controller re-reads the object poses and supplies robot timing.

For the no-human arm, omit `--plans` and generate 771 demonstrations into a different directory. For the public-data arm, acquire EgoDex separately, set `EGODEX_ROOT` to its local directory, process its hands and use `--styled`. The public-data selection and original held-out episode indices are in `fruitbowl/egodex.py` and the script defaults. Do not mix that development split with the six own-recording clip identifiers.

## 5. Replay the styles on three robots

```bash
python -m mani_skill.utils.download_asset xarm6 -y
python -m mani_skill.utils.download_asset widowxai -y
mkdir -p artifacts/robots
python scripts/replay_robots.py data/session2 \
  --robots panda xarm6_robotiq widowxai --random 128 --steps 260 \
  --video artifacts/robots --out artifacts/replay-robots.json
```

This is an object-aware controller experiment, not learned-policy evaluation. Fourteen source clips start with fruit already in the bowl and receive sampled replacement layouts, as described in [`results.md`](results.md). The `my_layouts` field in the original result records is a historical name for this mixed recorded-style set, not a claim of 30 exact measured layouts.

## 6. World-model training and fidelity analysis

Generate train/test transitions from disjoint clip styles, with action drift and gripper-command flips. The commands below match the saved generation recipe. The generator uses 512 parallel environments and no RGB rendering.

```bash
mkdir -p data/fruitbowl/wm_mine
HELDOUT="IMG_5796 IMG_5798 IMG_5801 IMG_5809 IMG_5816 IMG_5819"
python scripts/styled_states.py --plans data/session2/processed/plans.pkl \
  --exclude $HELDOUT --rounds 12 --noise 0 0.01 0.02 0.03 --grip-flip 0.002 \
  --out data/fruitbowl/wm_mine/train.npz
python scripts/styled_states.py --plans data/session2/processed/plans.pkl \
  --clips $HELDOUT --rounds 4 --noise 0 0.01 0.02 0.03 --grip-flip 0.002 \
  --seed 900000 --out data/fruitbowl/wm_mine/test.npz
python scripts/train_wm.py --data data/fruitbowl/wm_mine \
  --epochs 30 --horizon 8 --lr 3e-4 \
  --out data/fruitbowl/wm_mine/retrained-model.pt --results artifacts/worldmodel-retrained.json
python scripts/wm_fidelity.py --data data/fruitbowl/wm_mine \
  --model data/fruitbowl/wm_mine/model.pt --episodes 600 --horizon 40 \
  --name wm_fidelity_rerun
```

The fidelity command uses the **released** model (`model.pt`); change `--model` to evaluate the retrained one. It writes `results/wm_fidelity_rerun.*`. Do not overwrite the saved submission records. The original world-model initialization was not explicitly seeded, so retraining is not bit-exact. `--fruit-context` is an EgoDex-specific experimental option and was not used for the own-data model.

The trusted horizon is an empirical **median positional-error threshold**, not a guarantee for every rollout or a certificate that model-based control is safe. The full-episode score and rolling-origin fidelity score have different evaluation sets and must not be conflated.

## 7. Process the original or new phone recordings (optional)

The submitted own-data pipeline is `scripts/process_clips.py`: `<session>/raw/*.MOV` plus `<session>/targets.json` mapping clip stems to `persimmon`, `pear` or `mandarin`. A fixed board is visible in each clip. The target labels are manual; no audio is required.

```bash
python -m pip install -r requirements-tracking.txt
python scripts/process_clips.py data/session2 \
  --heldout IMG_5796 IMG_5798 IMG_5801 IMG_5809 IMG_5816 IMG_5819
```

The command above reprocesses the released recordings using the original split. For a new session, provide its own recordings, labels and held-out identifiers:

```bash
python scripts/process_clips.py data/my-session \
  --heldout heldout_clip_1 heldout_clip_2
```

WiLoR-mini and OWLv2 download their own model assets and may require accepting upstream model terms. Their model/data assets are not in this repository. `FRUITBOWL_MODELS` overrides the default `data/models` location for the optional MediaPipe comparison backend.

**Measure the printed board rather than assuming its designed size.** `fruitbowl/perception.py` currently encodes my measured print scale (0.906) and margin (73 mm). Update those constants for a different print/placement before processing new recordings. The session-level naming prototypes and workspace recentering use the session's clips; reserve an independent session if evaluating end-to-end perception generalization. Raw phone videos can consume substantial RAM during decoding.

`scripts/process_session.py` is the alternate protocol with a separate `calib.mp4`, optional Whisper transcription and `--heldout-clips`. It is not the protocol used for the reported 30 clips. The printed board is in `docs/assets/`.

## 8. ACT, Diffusion Policy and SmolVLA development baselines

The original baseline environment uses Python 3.11, LeRobot 0.4.4, PyTorch 2.10.0 + CUDA 12.8, torchvision 0.25.0 and SAPIEN 3.0.3. This differs from the core/tracking environment. Inference timings were measured in this baseline environment, including the FiLM model.

```bash
python3.11 -m venv .venv-baselines
source .venv-baselines/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-baselines.txt
python -m pip install -e . --no-deps
python scripts/to_lerobot.py --demos data/fruitbowl/demos_styled \
  --repo-id local/fruitbowl_styled128t --root data/fruitbowl/lerobot/styled128t \
  --size 128 --target-state
lerobot-train --policy.type=act --policy.chunk_size=20 --policy.n_action_steps=10 \
  --dataset.repo_id=local/fruitbowl_styled128t --dataset.root=data/fruitbowl/lerobot/styled128t \
  --batch_size=32 --steps=30000 --seed=0 --num_workers=8 --save_freq=10000 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --output_dir=data/fruitbowl/baselines/act
lerobot-train --policy.type=diffusion \
  --dataset.repo_id=local/fruitbowl_styled128t --dataset.root=data/fruitbowl/lerobot/styled128t \
  --batch_size=64 --steps=30000 --seed=0 --num_workers=8 --save_freq=10000 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --output_dir=data/fruitbowl/baselines/diffusion
python scripts/eval_policy.py --ckpt data/fruitbowl/baselines/act/checkpoints/last/pretrained_model \
  --target-state --size 128 --cam-keys observation.images.base,observation.images.wrist \
  --episodes 150 --seed 10000 --out artifacts/act-random.json
```

For Diffusion evaluation change the checkpoint path to `baselines/diffusion`. The saved training configuration files are in `configs/`. Baseline checkpoints and EgoDex footage are not bundled; these commands require regenerating the development demos.

SmolVLA uses a separate export **without `--target-state`**, camera keys renamed from `base`/`wrist` to `camera1`/`camera2`, and the public `lerobot/smolvla_base` initialization. The saved development results are from 30k steps, batch size 64, with ten actions executed per chunk. No completed own-data SmolVLA checkpoint is claimed. Training a VLA is not required to inspect or reproduce the submission's main own-data FiLM result.

## Reproducibility boundaries

The core task smoke test, released-policy smoke test, CPU tests and packaging checks were rerun for submission. The historical full training jobs and all 150-episode comparisons were **not** rerun at submission time. The retained JSON records are the evidence for those experiments. Package versions are pinned to the two recorded environments, but this is not a guarantee of compatibility with every GPU, operating system or future package index.
