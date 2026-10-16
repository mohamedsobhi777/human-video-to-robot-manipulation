# Human Video to Robot Manipulation

I recorded 30 phone videos of fruit being placed into a bowl, extracted key moments of the hand motion, and used them to generate robot demonstrations in simulation. A small image-based policy learns to move the requested fruit.

**[Download the PowerPoint — includes demo videos](docs/presentation.pptx?raw=true)** · [View slides as PDF](docs/presentation.pdf) · [Setup and reproduction](docs/reproduction.md)

## Watch it

[![My phone recording beside replay on Panda, xArm6 and WidowX](docs/figures/one-clip-three-robots.jpg)](docs/media/one-clip-three-robots.mp4)

- [One human recording → three robot arms](docs/media/one-clip-three-robots.mp4) — controller replay, not learned-policy transfer.
- [Policy trained from my recordings](docs/media/own-data-policy.mp4) — closed-loop manipulation in simulation.
- [World-model predictions vs. simulation](docs/media/world-model-imagination.mp4).

## Main result

The policy trains on **771 successful simulated demonstrations derived from 24 recordings**; six recordings are held out.

| Training demonstrations | Random layouts | Held-out layouts | Crowded layouts |
|---|---:|---:|---:|
| Scripted expert only | 90% | 84% | 85.3% |
| **Expert guided by my recordings** | **96%** | **86%** | **88%** |

Each cell is 150 evaluation episodes from one training seed. Held-out evaluation repeats six recorded layouts with reset variation; it is not 150 independent scenes. These are exploratory results, not a statistically established improvement. [Full results, baselines and caveats →](docs/results.md)

## How it works

1. **Video → keyframes.** Calibrate the table with a printed board, track the hand, and extract approach, grasp, carry and release parameters.
2. **Keyframes → robot demonstrations.** A closed-loop controller adapts those parameters to the robot and new object layouts. It uses simulator object poses.
3. **Demonstrations → learned policy.** A FiLM-conditioned policy uses two camera views, robot state and the selected fruit identity—not object poses—to predict action chunks.

Additional experiments cover three-arm controller replay and a short-horizon dynamics model. Their protocols and limitations are in the [results guide](docs/results.md).

## Run it and get the data

- **[Setup and reproduction guide](docs/reproduction.md)** — environment setup, pretrained evaluation, training and video processing. Core setup: Linux, NVIDIA GPU with Vulkan, Python 3.10 and FFmpeg. Evaluating the pretrained policy does not require the training videos.
- **[All 30 original recordings + target labels (205 MB ZIP)](https://github.com/mohamedsobhi777/human-video-to-robot-manipulation/releases/download/submission-v1/session2-recordings.zip)** — original video and audio preserved.
- **[Models and processed motion plans](https://github.com/mohamedsobhi777/human-video-to-robot-manipulation/releases/tag/submission-v1)** — FiLM policy, world-model checkpoint and all 30 plans; [SHA-256 checksums](docs/checkpoints.sha256).
- **Code:** [task](fruitbowl/env.py), [video processing](scripts/process_clips.py), [controller](fruitbowl/expert.py), [policy](scripts/image_bc.py), [world model](fruitbowl/worldmodel.py). [Saved experiment records](results/).

## Scope

Simulation only, one person, one recording session and three fruit identities. Language is a keyword/synonym parser, not open-vocabulary understanding. Perception calibration is shared across the session; there is no independent end-to-end perception test or real-robot result. The own-data SmolVLA run did not finish. [Other unsuccessful directions and limitations →](docs/results.md#failed-or-unfinished-work)

## Built on

[FiLM](https://arxiv.org/abs/1709.07871), [BC-Z](https://arxiv.org/abs/2202.02005), [ManiSkill](https://github.com/haosulab/ManiSkill), [LeRobot / SmolVLA](https://github.com/huggingface/lerobot), [EgoDex](https://github.com/apple/ml-egodex), [WiLoR-mini](https://github.com/warmshao/WiLoR-mini), [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) and [YCB](https://www.ycbbenchmarks.com/). Third-party assets retain their respective terms and are downloaded separately.
