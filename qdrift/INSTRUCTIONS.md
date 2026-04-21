# Q-Drifting: Instructions

## Overview

Q-Drifting is an **offline RL** algorithm that trains a policy capable of
generating high-quality actions in a **single forward pass** (no iterative ODE
solve at inference time).  It is built around a *drifting field* that combines
three forces:

| Force | Formula | Purpose |
|-------|---------|---------|
| Attraction | `V_attract = bc_actor(obs) − x_gen` | Anchor to behavioral distribution |
| Repulsion  | `V_repulse = Σ w_ij · (x_gen_j − x_gen_i)` | Prevent mode collapse |
| Q-gradient | `V_q = α · ∇Q(x_gen)` | RL improvement signal |

The policy internalises `V = V_attract − V_repulse + V_q` into its weights
during training, so at inference it just runs one forward pass and returns an
action.

### Two-phase training

**Phase 1 — BC Pretraining** (`bc_pretrain_steps`, default 200k)

Trains `bc_actor` (obs → action, no noise) via MSE loss on offline dataset
actions.  Goal: learn a smooth, generalized behavioral prior that interpolates
sensibly to unseen observations.  bc_actor is frozen after this phase.

**Phase 2 — Offline RL** (`offline_steps`, default 1M)

Trains `drifting_actor` (obs, noise → action) and the critic ensemble jointly.
The frozen `bc_actor` provides the attraction target in V_attract.
The Q-gradient steers actions toward higher-value regions on top of that prior.

---

## Setup

### Install dependencies

```bash
conda create -n qdrift python=3.11 -y
conda activate qdrift
pip install -r requirements.txt
```

On an HPC cluster, the SLURM template loads modules automatically:
```bash
module load GCCcore/12.3.0 Anaconda3 CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1
source activate qdrift
```

### Environment variables

```bash
export MUJOCO_GL=egl     # headless servers (HPC compute nodes)
export MUJOCO_GL=glfw    # local machines with a display

export OGBENCH_DATA_DIR="/path/to/ogbench_data"  # pre-downloaded datasets (cluster)
# Omit OGBENCH_DATA_DIR locally — OGBench will download to ~/.ogbench/data
```

The SLURM template (`slurm/train.slurm`) sets both variables automatically.

---

## Running Locally

### Debug mode (quick sanity check, ~1-2 min)

```bash
python main.py --debug --env_name=antmaze-medium-navigate-v0
```

Runs 100 BC steps + 100 offline steps + 2 eval episodes.

### Full local run

```bash
python main.py \
    --agent=agents/qdrift.py \
    --env_name=antmaze-large-navigate-v0 \
    --seed=0 \
    --run_group=qdrift_local
```

### Skip BC pretraining (for ablation)

```bash
python main.py --bc_pretrain_steps=0 --env_name=antmaze-medium-navigate-v0
```

---

## Submitting to an HPC Cluster

### Submission script

```bash
# All environments, seeds 0-4
bash scripts/submit_qdrift.sh

# Dry run (print commands without submitting)
bash scripts/submit_qdrift.sh --dry-run

# Debug jobs (30 min, 100+100 steps)
bash scripts/submit_qdrift.sh --debug

# Single environment
bash scripts/submit_qdrift.sh --env antmaze-large-navigate-v0

# Custom seeds and run group
bash scripts/submit_qdrift.sh --seeds "0 1 2" --run-group qdrift_v2

# Pass extra hyperparameters
bash scripts/submit_qdrift.sh --env antmaze-large-navigate-v0 -- \
    --agent.q_drift_scale=0.3 --agent.num_drift_samples=16
```

### Submission script flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | off | Print sbatch commands without submitting |
| `--debug` | off | Short trial jobs (30 min) |
| `--env ENV` | all | Only submit this environment |
| `--seeds "S"` | `"0 1 2 3 4"` | Space-separated seed list |
| `--run-group NAME` | `qdrift` | Name for organising results |
| `--` | | Everything after `--` is passed to `main.py` |

---

## Output Files

All files are written to `exp/<run_group>/<env_name>/<hash>/`.

```
exp/<run_group>/<env_name>/<hash>/
├── flags.json               — all CLI flags + agent hyperparameters
├── experiment_info.json     — system info, dataset shape, param count, git hash
├── bc_agent.csv             — Phase 1 training metrics (per log_interval steps)
├── offline_agent.csv        — Phase 2 training metrics (per log_interval steps)
├── eval.csv                 — evaluation results (per eval_interval steps)
├── inference_benchmark.json — standalone latency benchmark: single-obs, batch,
│                              and K-step comparison (run once after training)
├── results_summary.json     — best/final eval score, wall time, headline inference numbers
└── token.tk                 — written on clean completion (job deduplication)
```

### CSV columns

Every CSV has `step` and `wall_time` as the first two columns.

**bc_agent.csv**

| Column | Description |
|--------|-------------|
| `bc/loss` | MSE loss on dataset actions (primary BC metric; should decrease) |
| `bc/action_err` | Mean absolute error between predicted and dataset actions |
| `bc/action_err_max` | Max per-dimension error (worst-case prediction quality) |
| `bc/pred_action_norm` | L2 norm of predicted actions (should match dataset scale) |
| `bc/dataset_action_norm` | L2 norm of dataset actions (reference scale) |
| `bc/valid_fraction` | Fraction of valid (non-terminal) samples in the batch |
| `grad/norm`, `grad/max`, `grad/min` | Gradient statistics (from flax_utils) |

**offline_agent.csv**

| Column | Description |
|--------|-------------|
| `critic/critic_loss` | Bellman TD error (should decrease over training) |
| `critic/q_mean`, `q_max`, `q_min` | Live critic Q-value statistics |
| `critic/q_std` | Mean ensemble spread — high values = high uncertainty |
| `critic/td_error_mean` | Mean absolute TD error (convergence indicator) |
| `critic/td_error_max` | Maximum TD error in the batch |
| `critic/target_q_mean` | Mean bootstrap target Q (sanity-check scale) |
| `critic/target_q_max`, `target_q_min` | Target Q range |
| `batch/reward_mean`, `batch/reward_std` | Batch reward statistics |
| `batch/mask_mean` | Fraction of non-terminal transitions in batch |
| `actor/drift_loss` | Actor regression loss (should decrease) |
| `actor/drift_magnitude` | L2 norm of combined drifting field V |
| `actor/v_attract_mag` | Attraction force magnitude (‖V_attract‖) |
| `actor/v_repulse_mag` | Repulsion force magnitude (‖V_repulse‖); 0 if K=1 |
| `actor/v_q_mag` | Q-gradient force magnitude (‖α·∇Q‖) |
| `actor/q_attract_ratio` | `v_q_mag / v_attract_mag` — Q guidance vs BC anchoring |
| `actor/gen_action_mean` | Mean absolute generated action value |
| `actor/gen_action_std` | Std of generated actions (diversity measure) |
| `actor/bc_action_norm` | L2 norm of bc_actor outputs (frozen policy check) |
| `actor/sq_err_mean` | Mean squared error before masking (field step size) |
| `grad/norm`, `grad/max`, `grad/min` | Gradient statistics |

**Key diagnostic patterns:**

| Symptom | Likely cause |
|---------|-------------|
| `actor/v_q_mag` >> `actor/v_attract_mag` | `q_drift_scale` too large; reduce it |
| `actor/v_repulse_mag` ≈ 0 | All K actions collapsed to same point (mode collapse) |
| `critic/q_std` increasing | Critic ensemble diverging; add layer norm or reduce LR |
| `critic/td_error_mean` not decreasing | Critic not learning; check batch size and LR |
| `actor/bc_action_norm` ≈ 0 | BC actor outputting near-zero actions (BC training issue) |
| `grad/norm` > 10 | Gradient explosion; `clip_grad=True` should cap at 1.0 |

**eval.csv**

For each metric (e.g. `success`, `episode_return`, `episode_length`,
`inference_latency_ms`), five columns are written:

| Column | Description |
|--------|-------------|
| `{key}` | Mean across evaluation episodes |
| `{key}_std` | Standard deviation (for ±-style error bars) |
| `{key}_min` | Minimum episode value |
| `{key}_max` | Maximum episode value |
| `{key}_median` | Median (robust to outlier episodes) |

`inference_latency_ms` is the mean wall-clock time per `sample_actions()` call
(using `jax.block_until_ready()` for accuracy).  This gives a time-aligned
view of how inference speed evolves throughout training.

**inference_benchmark.json** — written once after training completes:

```json
{
  "device": "NVIDIA A100-SXM4-80GB",
  "single_obs": {
    "latency_ms_mean": 0.42,
    "latency_ms_std": 0.03,
    "throughput_per_sec": 2380
  },
  "batch_throughput": {
    "1":   {"latency_ms": 0.42, "throughput_per_sec": 2380},
    "256": {"latency_ms": 2.10, "throughput_per_sec": 121904}
  },
  "kstep_comparison": {
    "1":  {"latency_ms_mean": 0.42, "qdrift_speedup": 1.0},
    "8":  {"latency_ms_mean": 2.77, "qdrift_speedup": 6.6},
    "16": {"latency_ms_mean": 5.48, "qdrift_speedup": 13.0},
    "32": {"latency_ms_mean": 10.9, "qdrift_speedup": 26.0}
  }
}
```

`kstep_comparison` runs K sequential forward passes through the same
`DriftingActor` network to simulate a K-step diffusion/flow-matching policy.
`qdrift_speedup` is how much faster Q-Drifting (1 step) is than the K-step
baseline — this is the headline number for the inference-time contribution.

### JSON files

**experiment_info.json** — written once at startup:
```json
{
  "timestamp_start": "2026-04-18T10:30:00",
  "git_hash": "a1b2c3d4",
  "env_name": "antmaze-large-navigate-v0",
  "seed": 0,
  "phases": {"bc_pretrain_steps": 200000, "offline_steps": 1000000},
  "platform": {"hostname": "...", "jax_version": "...", "jax_devices": [...]},
  "dataset": {"size": 1000000, "observation_dim": 29, "action_dim": 8},
  "network": {"param_count": 12500000, "modules": ["critic", "drifting_actor", "bc_actor"]},
  "hyperparameters": {"lr": 0.0003, "num_qs": 10, ...}
}
```

**results_summary.json** — written once on completion:
```json
{
  "timestamp_end": "2026-04-18T22:30:00",
  "total_wall_time_human": "12h 0m 0s",
  "avg_steps_per_sec": 27.8,
  "eval": {
    "metric": "success",
    "best":  {"value": 0.82, "std": 0.04, "step": 950000},
    "final": {"value": 0.80, "std": 0.05, "step": 1200000}
  }
}
```

### Loading results in Python

```python
import pandas as pd
import matplotlib.pyplot as plt
import json

# --- Learning curve with error band ---
eval_df = pd.read_csv('exp/qdrift/antmaze-large-navigate-v0/<hash>/eval.csv')
plt.plot(eval_df['step'], eval_df['success'], label='mean')
plt.fill_between(
    eval_df['step'],
    eval_df['success'] - eval_df['success_std'],
    eval_df['success'] + eval_df['success_std'],
    alpha=0.2,
)

# --- Training metrics ---
offline_df = pd.read_csv('...offline_agent.csv')
plt.plot(offline_df['wall_time'], offline_df['actor/drift_loss'])

# --- Final results ---
with open('.../results_summary.json') as f:
    summary = json.load(f)
print(f"Best success: {summary['eval']['best']['value']:.3f}")
```

---

## main.py Flags

### Core flags

| Flag | Default | Description |
|------|---------|-------------|
| `--env_name` | `antmaze-large-navigate-v0` | OGBench environment |
| `--seed` | `0` | Random seed |
| `--run_group` | `Debug` | Experiment group name (appears in save path) |
| `--save_dir` | `exp/` | Root directory for all output files |
| `--agent` | `agents/qdrift.py` | Agent config file |

### Training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bc_pretrain_steps` | `200000` | Phase 1: BC pretraining steps |
| `--offline_steps` | `1000000` | Phase 2: offline RL steps |
| `--horizon_length` | `1` | N-step return horizon (keep at 1) |

### Logging and evaluation

| Flag | Default | Description |
|------|---------|-------------|
| `--log_interval` | `5000` | Write to CSV every N steps |
| `--eval_interval` | `50000` | Run evaluation every N steps |
| `--save_interval` | `50000` | Save checkpoint every N steps (0=never) |
| `--eval_episodes` | `50` | Episodes per evaluation checkpoint |
| `--video_episodes` | `0` | Extra rendered episodes (not counted in stats) |

### Dataset flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_proportion` | `1.0` | Fraction of dataset to use (ablations) |
| `--ogbench_dataset_dir` | `None` | Pre-downloaded dataset directory |
| `--sparse` | `False` | Convert to sparse rewards (-1/0) |

### Special flags

| Flag | Default | Description |
|------|---------|-------------|
| `--debug` | `False` | Short run: 100+100 steps, 2 eval episodes |
| `--small` | `False` | CPU-friendly preset (see below) |
| `--auto_cleanup` | `True` | Delete checkpoint files on completion |

---

## Agent Hyperparameters

Pass via `--agent.PARAM=VALUE`.

### Drifting field

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_drift_samples` | `8` | K: generated actions per state (for repulsion) |
| `drift_temperature` | `1.0` | T: repulsion kernel temperature |
| `q_drift_scale` | `0.1` | α: weight on Q-gradient term |

### RL

| Parameter | Default | Description |
|-----------|---------|-------------|
| `discount` | `0.99` | γ — use `0.995` for giant mazes |
| `num_qs` | `10` | Critic ensemble size |
| `rho` | `0.5` | Pessimism coefficient |
| `tau` | `0.005` | Target network EMA rate |
| `best_of_n` | `1` | Best-of-N at inference (1 = no resampling) |

### Network architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `3e-4` | Adam learning rate |
| `batch_size` | `256` | Training batch size |
| `actor_hidden_dims` | `(512,512,512,512)` | Drifting actor MLP dims |
| `value_hidden_dims` | `(512,512,512,512)` | Critic MLP dims |
| `bc_hidden_dims` | `(512,512,512,512)` | BC policy MLP dims |
| `actor_layer_norm` | `False` | Layer norm in drifting actor |
| `value_layer_norm` | `True` | Layer norm in critic |
| `bc_layer_norm` | `False` | Layer norm in BC policy |
| `clip_grad` | `True` | Clip global gradient norm to 1.0 |
| `use_target_grad` | `True` | Use target critic for Q-gradient |

---

## CPU vs GPU: Hardware Requirements

### Why the default config needs a GPU

The default configuration was designed for GPU throughput:

| Bottleneck | Default | Cost driver |
|---|---|---|
| Critic ensemble | `num_qs=10` | 10 parallel Q-networks, dominates step time |
| Network width | `4 × 512` hidden dims | ~3.4M trainable params |
| Drift samples | `K=8` per state | 8× critic forward passes for Q-gradient |
| Training budget | 1.2M total steps | Needed to converge on hard mazes |

Each training step costs **~16 GFLOPs**. On a modern GPU (~10 TFLOPS) this
takes ~2–5 ms → **~30 min total**. On CPU (~50–100 GFLOPS effective) this takes
~200 ms → **~55–110 hours total**. Not feasible for local iteration.

At training start, the script runs a timing warmup and prints a concrete ETA
for your hardware so you know immediately whether to use `--small`.

### `--small` preset (CPU-viable)

The `--small` flag applies a coordinated reduction across the three main cost
drivers, keeping the algorithm scientifically intact:

| Parameter | Default | `--small` | Speedup |
|---|---|---|---|
| `num_qs` | 10 | 2 | ~5× (biggest win) |
| Hidden dims | `4 × 512` | `3 × 256` | ~8× fewer params |
| Drift samples K | 8 | 4 | ~2× cheaper actor loss |
| Batch size | 256 | 128 | ~2× cheaper per step |
| `bc_pretrain_steps` | 200k | 100k | 2× fewer |
| `offline_steps` | 1M | 300k | 3× fewer |

Combined: **~12× cheaper per step**, **~3× fewer steps** → roughly **36× faster
overall**. Expected wall time: **3–6 hours on a modern laptop CPU**.

```bash
# CPU-local run
python main.py --small --env_name=antmaze-medium-navigate-v0 --seed=0

# Debug first to check env loads correctly (< 2 min)
python main.py --debug --env_name=antmaze-medium-navigate-v0

# Full GPU run (HPC cluster)
bash scripts/submit_qdrift.sh --env antmaze-medium-navigate-v0 --seeds "0 1 2"
```

`num_qs=2` is the standard minimum for pessimistic TD (same as IQL, TD3+BC).
The algorithm remains valid; it just has less pessimism than the GPU config.
Results from `--small` runs are useful for ablations and qualitative comparisons
but should not be reported as the primary numbers in a paper — use GPU results
for those.

## Environments

### Antmaze (navigation)

| Environment | Difficulty | Notes |
|-------------|-----------|-------|
| `antmaze-medium-navigate-v0` | Easy | Start here |
| `antmaze-large-navigate-v0` | Medium | Standard benchmark |
| `antmaze-giant-navigate-v0` | Hard | Use `discount=0.995` |
| `antmaze-medium-stitch-v0` | Medium | Requires trajectory stitching |
| `antmaze-large-stitch-v0` | Hard | Requires trajectory stitching |
| `antmaze-giant-stitch-v0` | Very hard | Use `discount=0.995` |

### Humanoidmaze (high-dimensional navigation)

| Environment | Difficulty | Notes |
|-------------|-----------|-------|
| `humanoidmaze-medium-navigate-v0` | Hard | Use `discount=0.995` |
| `humanoidmaze-large-navigate-v0` | Very hard | Use `discount=0.995` |
| `humanoidmaze-giant-navigate-v0` | Extreme | Use `discount=0.995` |
| `humanoidmaze-medium-stitch-v0` | Very hard | Use `discount=0.995` |
| `humanoidmaze-large-stitch-v0` | Extreme | Use `discount=0.995` |
| `humanoidmaze-giant-stitch-v0` | Extreme | Use `discount=0.995` |

---

## Recommended Environments for Testing the Thesis

Q-Drifting's thesis has two components: **(1) task performance** (offline RL quality)
and **(2) inference speed** (1-step >> K-step).  Different environments stress-test
each component differently.

### For Task Performance (does the algorithm actually work?)

| Priority | Environment | Why |
|----------|-------------|-----|
| ★★★ | `antmaze-medium-navigate-v0` | Standard benchmark; fast to evaluate; converges in ~30 min GPU |
| ★★★ | `antmaze-large-navigate-v0` | The de-facto offline RL navigation benchmark |
| ★★  | `antmaze-medium-stitch-v0` | Tests trajectory stitching — requires RL, not just BC |
| ★★  | `antmaze-large-stitch-v0` | Hard stitching; separates RL methods from BC-only baselines |
| ★   | `antmaze-giant-navigate-v0` | Long-horizon; use `--agent.discount=0.995` |
| ★   | `humanoidmaze-medium-navigate-v0` | High-dim obs (378-dim); use `--agent.discount=0.995` |

**Start here for quick validation:**
```bash
python main.py --debug --env_name=antmaze-medium-navigate-v0   # 2 min sanity check
python main.py --small --env_name=antmaze-medium-navigate-v0   # 3-6 h CPU run
```

### For Inference Speed (the core novelty claim)

The inference benchmark runs automatically after training on **any** environment.
Use `antmaze-medium-navigate-v0` (fastest training) to get numbers quickly, then
confirm on `antmaze-large-navigate-v0` for the paper.

The key table to report:
```
K-step comparison (same network, K sequential forward passes):
  K=1  : baseline (Q-Drifting)
  K=8  : typical diffusion policy
  K=16 : DDPM-style policy
  K=32 : high-quality diffusion
```
These are produced automatically in `inference_benchmark.json` and
`results_summary.json`.

### Recommended HPC Submission Commands

```bash
# === Core paper results (antmaze navigate + stitch, 5 seeds each) ===
bash scripts/submit_qdrift.sh \
    --env antmaze-medium-navigate-v0 \
    --seeds "0 1 2 3 4" \
    --run-group qdrift_v1

bash scripts/submit_qdrift.sh \
    --env antmaze-large-navigate-v0 \
    --seeds "0 1 2 3 4" \
    --run-group qdrift_v1

bash scripts/submit_qdrift.sh \
    --env antmaze-medium-stitch-v0 \
    --seeds "0 1 2 3 4" \
    --run-group qdrift_v1

bash scripts/submit_qdrift.sh \
    --env antmaze-large-stitch-v0 \
    --seeds "0 1 2 3 4" \
    --run-group qdrift_v1

# === Giant mazes (higher discount) ===
bash scripts/submit_qdrift.sh \
    --env antmaze-giant-navigate-v0 \
    --seeds "0 1 2 3 4" \
    --run-group qdrift_v1 \
    -- --agent.discount=0.995

bash scripts/submit_qdrift.sh \
    --env antmaze-giant-stitch-v0 \
    --seeds "0 1 2 3 4" \
    --run-group qdrift_v1 \
    -- --agent.discount=0.995

# === Or submit everything at once (all 12 environments, 5 seeds = 60 jobs) ===
bash scripts/submit_qdrift.sh --run-group qdrift_v1

# === Ablation: effect of q_drift_scale ===
for SCALE in 0.01 0.05 0.1 0.3 1.0; do
    bash scripts/submit_qdrift.sh \
        --env antmaze-large-navigate-v0 \
        --seeds "0 1 2" \
        --run-group qdrift_abl_qdscale \
        -- --agent.q_drift_scale=${SCALE}
done

# === Ablation: effect of num_drift_samples (K) ===
for K in 1 2 4 8 16; do
    bash scripts/submit_qdrift.sh \
        --env antmaze-large-navigate-v0 \
        --seeds "0 1 2" \
        --run-group qdrift_abl_K \
        -- --agent.num_drift_samples=${K}
done

# === Debug job (30 min, to test the pipeline end-to-end) ===
bash scripts/submit_qdrift.sh \
    --debug \
    --env antmaze-medium-navigate-v0 \
    --seeds "0"
```

---

## Suggested Hyperparameter Sweeps

The three most impactful hyperparameters:

```bash
# Q-gradient scale (most important for RL improvement)
--agent.q_drift_scale=0.01
--agent.q_drift_scale=0.1    # default
--agent.q_drift_scale=0.3
--agent.q_drift_scale=1.0

# Repulsion kernel temperature
--agent.drift_temperature=0.1
--agent.drift_temperature=0.5
--agent.drift_temperature=1.0  # default
--agent.drift_temperature=5.0

# Generated actions per state (more = better repulsion, slower training)
--agent.num_drift_samples=1    # no repulsion (ablation)
--agent.num_drift_samples=4
--agent.num_drift_samples=8    # default
--agent.num_drift_samples=16

# BC pretraining budget
--bc_pretrain_steps=50000
--bc_pretrain_steps=200000     # default
--bc_pretrain_steps=500000
```

---

## Monitoring Training

Key metrics to watch in `offline_agent.csv`:

| Metric | What to expect |
|--------|---------------|
| `actor/drift_loss` | Should decrease steadily throughout training |
| `actor/drift_magnitude` | How far actions move per step; should decrease |
| `actor/v_q_mag` | Q-gradient magnitude; large values (>10) indicate instability |
| `actor/v_attract_mag` | Attraction force; should be reasonable relative to v_q_mag |
| `critic/q_mean` | Q-value estimates; should increase as policy improves |
| `grad/norm` | Gradient norm; spikes indicate instability |

Key metrics in `eval.csv`:

| Metric | Notes |
|--------|-------|
| `success` | Navigation task success rate (0–1), the primary metric |
| `success_std` | Standard deviation across eval episodes; use for error bars |
| `episode_length` | Steps per episode; shorter = more direct navigation |
| `episode_return` | Cumulative reward (correlated with `success` for dense rewards) |

---

## Project Structure

```
qdrift/
├── agents/
│   ├── __init__.py          — agent registry
│   └── qdrift.py            — QDriftAgent: critic, BC, drifting actor, training
├── envs/
│   ├── env_utils.py         — environment wrappers, D4RL loading
│   ├── ogbench_utils.py     — OGBench dataset loading
│   └── d4rl_utils.py        — D4RL benchmark utilities (legacy)
├── utils/
│   ├── networks.py          — MLP, Value, BCPolicy, DriftingActor
│   ├── flax_utils.py        — TrainState, ModuleDict, save/restore
│   └── datasets.py          — Dataset, ReplayBuffer
├── main.py                  — training entry point (two-phase)
├── evaluation.py            — episode rollouts + statistics
├── log_utils.py             — CsvLogger, ExperimentLogger, JSON metadata
├── slurm/train.slurm        — SLURM job template (HPC cluster)
├── scripts/submit_qdrift.sh — batch submission script
├── requirements.txt         — Python dependencies
└── INSTRUCTIONS.md          — this file
```
