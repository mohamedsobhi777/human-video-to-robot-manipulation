"""Inference speed of the policies on one GPU: latency per call and per executed action.

Each policy predicts a chunk of actions per call and executes some of them before
calling again (FiLM: 10 predicted, 5 executed; SmolVLA: 50/10; ACT: 20/10;
Diffusion Policy: 16/8). Time per executed action = call latency / actions executed.
Inputs are random tensors of the real shapes; weights are the trained checkpoints
(SmolVLA: the public base model, same architecture as the fine-tuned one).
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torchvision

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=100)
p.add_argument("--out", default="results/bench_inference.json")
p.add_argument("--data", default="data/fruitbowl", help="directory containing policy checkpoints")
args = p.parse_args()
dev = "cuda"
D = Path(args.data)


class FiLMResNet(nn.Module):        # as in image_bc.py
    def __init__(self, emb_dim):
        super().__init__()
        r = torchvision.models.resnet18()
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.stages = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.film = nn.ModuleList([nn.Linear(emb_dim, 2*c) for c in (64, 128, 256, 512)])

    def forward(self, x, emb):
        x = self.stem(x)
        for stage, film in zip(self.stages, self.film):
            g, b = film(emb).chunk(2, -1)
            x = stage(x)*(1 + g[..., None, None]) + b[..., None, None]
        return x.mean((2, 3))


class FiLMPolicy(nn.Module):
    def __init__(self, chunk=10):
        super().__init__()
        self.chunk = chunk
        self.base, self.wrist = FiLMResNet(64), FiLMResNet(64)
        self.fruit = nn.Embedding(3, 64)
        self.head = nn.Sequential(nn.Linear(512*2 + 6 + 64, 1024), nn.SiLU(), nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, 6*chunk))
        for n in ("mean", "std"): self.register_buffer(n, torch.zeros(1, 3, 1, 1))
        for n in ("s_m", "s_s", "a_m", "a_s"): self.register_buffer(n, torch.zeros(6))

    def forward(self, base, wrist, state, target):
        emb = self.fruit(target)
        norm = lambda x: (x.float()/255 - self.mean)/self.std
        z = torch.cat([self.base(norm(base), emb), self.wrist(norm(wrist), emb), (state - self.s_m)/self.s_s, emb], 1)
        return self.head(z).view(-1, self.chunk, 6)*self.a_s + self.a_m


def timeit(fn, iters):
    for _ in range(10): fn()
    torch.cuda.synchronize(); times = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    return float(np.median(times)*1000), float(np.percentile(times, 90)*1000)


results = {}
params = lambda m: sum(p.numel() for p in m.parameters())

# --- FiLM ResNet-18 (ours): fp32, then fp16 + channels_last + torch.compile ------------
film = FiLMPolicy().to(dev).eval()
film.load_state_dict(torch.load(D/"film_session2.pt", map_location=dev, weights_only=True))
for B in (1, 64):
    x = dict(base=torch.randint(0, 255, (B, 3, 112, 112), device=dev, dtype=torch.uint8),
             wrist=torch.randint(0, 255, (B, 3, 96, 96), device=dev, dtype=torch.uint8),
             state=torch.randn(B, 6, device=dev), target=torch.randint(0, 3, (B,), device=dev))
    with torch.no_grad():
        results[f"film_fp32_b{B}"] = dict(params=params(film), executed=5, ms=timeit(lambda: film(**x), args.iters))
fast = FiLMPolicy().to(dev).eval()
fast.load_state_dict(film.state_dict())
fast = torch.compile(fast.to(memory_format=torch.channels_last).half(), mode="max-autotune-no-cudagraphs")
for B in (1, 64):
    x = dict(base=torch.randint(0, 255, (B, 3, 112, 112), device=dev, dtype=torch.uint8).contiguous(memory_format=torch.channels_last),
             wrist=torch.randint(0, 255, (B, 3, 96, 96), device=dev, dtype=torch.uint8).contiguous(memory_format=torch.channels_last),
             state=torch.randn(B, 6, device=dev).half(), target=torch.randint(0, 3, (B,), device=dev))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        results[f"film_fp16_compiled_b{B}"] = dict(params=params(film), executed=5, ms=timeit(lambda: fast(**x), args.iters))
print({k: v["ms"] for k, v in results.items()}, flush=True)

# --- lerobot policies -------------------------------------------------------------
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


def lerobot_bench(name, ckpt, batch_fn, executed):
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    pol = get_policy_class(cfg.type).from_pretrained(ckpt).to(dev).eval()
    batch = batch_fn(pol)
    with torch.no_grad():
        # reset + select_action always runs one full chunk prediction (the queues start empty)
        results[name] = dict(params=params(pol), executed=executed, ms=timeit(lambda: (pol.reset(), pol.select_action(batch)), max(20, args.iters//5)))
    print(name, results[name], flush=True)
    del pol; torch.cuda.empty_cache()


img = lambda s: torch.rand(1, 3, s, s, device=dev)
lerobot_bench("act", str(D/"baselines/act/checkpoints/last/pretrained_model"),
              lambda pol: {"observation.images.base": img(128), "observation.images.wrist": img(128), "observation.state": torch.randn(1, 9, device=dev)}, 10)
lerobot_bench("diffusion", str(D/"baselines/diffusion/checkpoints/last/pretrained_model"),
              lambda pol: {"observation.images.base": img(128), "observation.images.wrist": img(128), "observation.state": torch.randn(1, 9, device=dev)}, 8)


def smolvla_batch(pol):
    from lerobot.policies.factory import make_pre_post_processors
    pre, _ = make_pre_post_processors(pol.config, pretrained_path="lerobot/smolvla_base")
    b = {k: img(256) for k in pol.config.image_features}
    b.update({"observation.state": torch.randn(1, 6, device=dev), "task": ["put the pear in the bowl"]})
    return pre(b)
lerobot_bench("smolvla", "lerobot/smolvla_base", smolvla_batch, 10)

for k, v in results.items():
    v["ms_per_executed_action"] = v["ms"][0]/v["executed"]
print(json.dumps(results, indent=1))
json.dump(results, open(args.out, "w"), indent=1)
