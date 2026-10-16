"""Small image policy baseline (local GPU): pretrained ResNet-18 on the over-shoulder
and wrist images + robot state + target fruit (keyword match on the instruction)
-> next 10 actions. Trained on the same demos as SmolVLA; evaluated closed-loop.
"""
import argparse, json, sys, time
from pathlib import Path
import imageio.v3 as iio, numpy as np, torch, torch.nn as nn, torchvision, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import rollout
from fruitbowl.env import FRUITS
from fruitbowl.perception import target_from_text

p = argparse.ArgumentParser()
p.add_argument("--demos", default="data/fruitbowl/demos_styled")
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--chunk", type=int, default=10)
p.add_argument("--execute", type=int, default=5)
p.add_argument("--episodes", type=int, default=128)
p.add_argument("--seed", type=int, default=50000)
p.add_argument("--layouts-file")
p.add_argument("--out", default="results/image_bc.json")
p.add_argument("--train-seed", type=int, default=0)
p.add_argument("--load", help="skip training; evaluate this saved policy")
p.add_argument("--paraphrase", action="store_true", help="instruction phrasings never seen in training")
p.add_argument("--video", help="mp4 of the first batch (first 4 envs)")
p.add_argument("--film", action="store_true", help="condition the image encoders on the target fruit (FiLM)")
p.add_argument("--save", default="data/fruitbowl/image_bc.pt")
args = p.parse_args()
if not 0 < args.execute <= args.chunk or args.episodes < 1:
    p.error("require 0 < --execute <= --chunk and --episodes >= 1")
dev = "cuda"
torch.manual_seed(args.train_seed); np.random.seed(args.train_seed)

# ---- data (decoded once into memory, 112x112 to keep it small) ----
if args.load:
    checkpoint = torch.load(args.load, map_location="cpu", weights_only=True)
    s_m, s_s, a_m, a_s = (checkpoint[key] for key in ("s_m", "s_s", "a_m", "a_s"))
    args.steps = 0
else:
    files = sorted(Path(args.demos).glob("ep_*.npz"))
    if not files:
        p.error(f"no ep_*.npz demonstrations found in {args.demos}")
    B, W, S, A, Tg = [], [], [], [], []
    small = lambda v, s: torch.nn.functional.interpolate(torch.tensor(v).permute(0, 3, 1, 2).float(), size=(s, s), mode="area").to(torch.uint8)
    for n, f in enumerate(files):
        d = np.load(f)
        base, wrist = iio.imread(f.with_name(f.stem + "_base.mp4")), iio.imread(f.with_name(f.stem + "_hand.mp4"))
        T = min(len(d["state"]), len(base), len(wrist))
        B.append(small(base[:T], 112)); W.append(small(wrist[:T], 96))
        S.append(torch.tensor(d["state"][:T]).float())
        a = d["action"][:T]; a = np.concatenate([a, np.repeat(a[-1:], args.chunk, 0)])
        A.append(torch.tensor(np.stack([a[t:t+args.chunk] for t in range(T)])).float())
        Tg.append(torch.full((T,), FRUITS.index(target_from_text(str(d["instruction"])))))
    B, W, S, A, Tg = map(torch.cat, (B, W, S, A, Tg))
    print(f"{len(files)} episodes, {len(S)} frames", flush=True)
    s_m, s_s = S.mean(0), S.std(0) + 1e-4
    a_m, a_s = A.reshape(-1, 6).mean(0), A.reshape(-1, 6).std(0) + 1e-4


class FiLMResNet(nn.Module):
    """ResNet-18 whose stage outputs are scaled and shifted by the target-fruit
    embedding (FiLM), so the instruction shapes what the vision stream attends to."""
    def __init__(self, emb_dim):
        super().__init__()
        r = torchvision.models.resnet18(weights=None if args.load else "IMAGENET1K_V1")
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.stages = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.film = nn.ModuleList([nn.Linear(emb_dim, 2*c) for c in (64, 128, 256, 512)])
        for f in self.film: nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)   # start as the plain ResNet

    def forward(self, x, emb):
        x = self.stem(x)
        for stage, film in zip(self.stages, self.film):
            g, b = film(emb).chunk(2, -1)
            x = stage(x)*(1 + g[..., None, None]) + b[..., None, None]
        return x.mean((2, 3))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.film = args.film
        def enc():
            if args.film: return FiLMResNet(64)
            r = torchvision.models.resnet18(weights=None if args.load else "IMAGENET1K_V1"); r.fc = nn.Identity(); return r
        self.base, self.wrist = enc(), enc()
        self.fruit = nn.Embedding(3, 64)
        self.head = nn.Sequential(nn.Linear(512*2 + 6 + 64, 1024), nn.SiLU(), nn.Linear(1024, 1024), nn.SiLU(),
                                  nn.Linear(1024, 6*args.chunk))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for k, v in dict(s_m=s_m, s_s=s_s, a_m=a_m, a_s=a_s).items(): self.register_buffer(k, v.clone())

    def forward(self, base, wrist, state, target):
        emb = self.fruit(target)
        norm = lambda x: (x.float()/255 - self.mean)/self.std
        f = (lambda x, e: e(norm(x), emb)) if self.film else (lambda x, e: e(norm(x)))
        z = torch.cat([f(base, self.base), f(wrist, self.wrist), (state - self.s_m)/self.s_s, emb], 1)
        return self.head(z).view(-1, args.chunk, 6)

    def act(self, base, wrist, state, target):
        return self(base, wrist, state, target)*self.a_s + self.a_m


pol = Policy().to(dev)
if args.load:
    pol.load_state_dict(checkpoint)
opt = torch.optim.AdamW(pol.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=max(args.steps, 1), pct_start=0.05)
def aug(x, pad=6):
    """Random shift augmentation: pad by `pad` pixels and crop back at a random offset per sample."""
    n, c, h, w = x.shape
    xp = torch.nn.functional.pad(x.float(), (pad, pad, pad, pad), mode="replicate")
    ox, oy = torch.randint(0, 2*pad + 1, (2, n))
    return torch.stack([xp[k, :, oy[k]:oy[k]+h, ox[k]:ox[k]+w] for k in range(n)]).to(torch.uint8)
t0 = time.time()
for step in range(args.steps):
    i = torch.randint(len(S), (128,))
    base = aug(B[i]).to(dev)
    loss = (pol(base, W[i].to(dev), S[i].to(dev), Tg[i].to(dev)) - ((A[i].to(dev) - pol.a_m)/pol.a_s)).abs().mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 2000 == 0: print(f"step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
if not args.load:
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), args.save)

# ---- closed-loop evaluation ----
pol.eval()
env = gym.make("FruitToBowl-v1", num_envs=min(args.episodes, 64), obs_mode="rgb", control_mode="pd_ee_pose",
               robot_uids="panda_wristcam", render_mode="rgb_array" if args.video else None, sim_backend="physx_cuda")
from fruitbowl.env import HELDOUT_TEMPLATES
frames = []
u = env.unwrapped
fixed = json.load(open(args.layouts_file)) if args.layouts_file else None
res = []
for b in range(0, args.episodes, u.num_envs):
    layouts = [dict(fixed[(b + j) % len(fixed)]) for j in range(u.num_envs)] if fixed else None
    if args.paraphrase:
        prng = np.random.default_rng(args.seed + b)
        layouts = layouts or [dict(target=int(prng.integers(3))) for _ in range(u.num_envs)]
        for l in layouts:
            l["instruction"] = HELDOUT_TEMPLATES[prng.integers(len(HELDOUT_TEMPLATES))].format(FRUITS[l["target"]])
    env.reset(seed=args.seed + b, options=dict(layouts=layouts) if layouts else {})
    rollout.settle(env)
    tgt = torch.tensor([FRUITS.index(target_from_text(s)) for s in u.instructions], device=dev)
    with torch.no_grad():
        for k in range(0, 200, args.execute):
            o = u.get_obs()["sensor_data"]
            im = lambda c, s: torch.nn.functional.interpolate(o[c]["rgb"].permute(0, 3, 1, 2).float(), size=(s, s), mode="area")
            chunk = pol.act(im("base_camera", 112), im("hand_camera", 96), rollout.robot_state(u).float(), tgt)
            for j in range(args.execute):
                for _ in range(2):
                    obs, r, term, trunc, info = env.step(rollout.to_env_action(chunk[:, j], u.device))
                if args.video and b == 0:
                    frames.append(np.concatenate(list(env.render()[:4].cpu().numpy()), 1))
    res += [dict(target=FRUITS[int(u.target[j])], success=bool(info["success"][j]), others_in_bowl=bool(info["others_in_bowl"][j]))
            for j in range(min(u.num_envs, args.episodes - b))]
summary = dict(episodes=len(res), success=float(np.mean([r["success"] for r in res])),
               per_fruit={f: float(np.mean([r["success"] for r in res if r["target"] == f] or [np.nan])) for f in FRUITS},
               wrong_fruit_in_bowl=float(np.mean([r["others_in_bowl"] for r in res])), demos=args.demos, steps=args.steps)
print(json.dumps(summary, indent=1))
summary.update(train_seed=args.train_seed, seed=args.seed, layouts_file=args.layouts_file, paraphrase=args.paraphrase, load=args.load)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(dict(summary=summary, episodes=res), open(args.out, "w"), indent=1)
if args.video:
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    import imageio; imageio.mimwrite(args.video, frames, fps=10)
env.close()
