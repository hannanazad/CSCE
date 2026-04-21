# Q-Drifting: Job Submission Commands

All commands below use `scripts/submit_qdrift.sh` and must be run from the
**project root** (`qdrift/`).  Set `OGBENCH_DATA_DIR` in `slurm/train.slurm`
before submitting.

Append `--dry-run` to any command to print the `sbatch` calls without
actually submitting them.

---

## 0. Quick sanity checks (run these first)

```bash
# Local debug run — 100+100 steps, 2 eval episodes (~2 min on any hardware)
python main.py --debug --env_name=antmaze-medium-navigate-v0 --seed=0

# Dry-run the submit script to confirm flags before touching the cluster
bash scripts/submit_qdrift.sh --dry-run --env antmaze-medium-navigate-v0 --seeds "0"

# Single debug SLURM job (30 min wall time, 100+100 steps)
bash scripts/submit_qdrift.sh --debug \
    --env antmaze-medium-navigate-v0 --seeds "0" --run-group debug
```

---

## 1. Core results  (3 seeds × all 12 environments)

These are the primary numbers for the paper.  Each job requests 12 h GPU.

### 1a. Antmaze — Navigate

```bash
bash scripts/submit_qdrift.sh \
    --env antmaze-medium-navigate-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1

bash scripts/submit_qdrift.sh \
    --env antmaze-large-navigate-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1

bash scripts/submit_qdrift.sh \
    --env antmaze-giant-navigate-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1 \
    -- --agent.discount=0.995
```

### 1b. Antmaze — Stitch

Stitch tasks require genuine trajectory composition (the Q-gradient must bridge
sub-optimal trajectory fragments), so they are a stronger test of the RL
improvement over pure BC.

```bash
bash scripts/submit_qdrift.sh \
    --env antmaze-medium-stitch-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1

bash scripts/submit_qdrift.sh \
    --env antmaze-large-stitch-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1

bash scripts/submit_qdrift.sh \
    --env antmaze-giant-stitch-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1 \
    -- --agent.discount=0.995
```

### 1c. Humanoidmaze — Navigate

High-dimensional observation space (378-dim).  Use `discount=0.995` throughout
because episodes are long.

```bash
bash scripts/submit_qdrift.sh \
    --env humanoidmaze-medium-navigate-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1 \
    -- --agent.discount=0.995

bash scripts/submit_qdrift.sh \
    --env humanoidmaze-large-navigate-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1 \
    -- --agent.discount=0.995

bash scripts/submit_qdrift.sh \
    --env humanoidmaze-giant-navigate-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1 \
    -- --agent.discount=0.995
```

### 1d. Humanoidmaze — Stitch

```bash
bash scripts/submit_qdrift.sh \
    --env humanoidmaze-medium-stitch-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1 \
    -- --agent.discount=0.995

bash scripts/submit_qdrift.sh \
    --env humanoidmaze-large-stitch-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1 \
    -- --agent.discount=0.995

bash scripts/submit_qdrift.sh \
    --env humanoidmaze-giant-stitch-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_v1 \
    -- --agent.discount=0.995
```

### 1e. Submit everything at once

```bash
bash scripts/submit_qdrift.sh --run-group qdrift_v1
# Note: giant environments automatically use discount=0.995 inside submit_qdrift.sh
```

---

## 2. Ablation — Q-gradient scale  (`q_drift_scale`, α)

Tests how strongly the Q-gradient steers away from the BC prior.
Use 3 seeds on a mid-difficulty environment.

| Value | Description |
|-------|-------------|
| `0.0` | Pure BC with repulsion (no RL signal) |
| `0.01` | Very light Q-guidance |
| `0.05` | Reduced Q-guidance |
| `0.1` | **Default** |
| `0.3` | Stronger Q-guidance |
| `1.0` | Aggressive Q-guidance (risk of instability) |

```bash
for SCALE in 0.0 0.01 0.05 0.1 0.3 1.0; do
    bash scripts/submit_qdrift.sh \
        --env antmaze-large-navigate-v0 \
        --seeds "0 1 2" \
        --run-group qdrift_abl_qdscale \
        -- --agent.q_drift_scale=${SCALE}
done
```

---

## 3. Ablation — Number of drift samples  (`num_drift_samples`, K)

Tests the effect of the repulsion term.  K=1 disables repulsion entirely.

| Value | Description |
|-------|-------------|
| `1` | No repulsion (BC-only drifting field) |
| `2` | Minimal repulsion |
| `4` | Reduced repulsion |
| `8` | **Default** |
| `16` | Dense repulsion (2× more compute per step) |

```bash
for K in 1 2 4 8 16; do
    bash scripts/submit_qdrift.sh \
        --env antmaze-large-navigate-v0 \
        --seeds "0 1 2" \
        --run-group qdrift_abl_K \
        -- --agent.num_drift_samples=${K}
done
```

---

## 4. Ablation — Repulsion temperature  (`drift_temperature`, T)

Controls the width of the repulsion kernel.  Low T → sharp repulsion between
nearby actions only.  High T → uniform repulsion across all K samples.

| Value | Description |
|-------|-------------|
| `0.1` | Sharp, local repulsion |
| `0.5` | Moderate |
| `1.0` | **Default** |
| `5.0` | Broad, near-uniform repulsion |
| `10.0` | Effectively uniform (all weights ≈ 1/K) |

```bash
for TEMP in 0.1 0.5 1.0 5.0 10.0; do
    bash scripts/submit_qdrift.sh \
        --env antmaze-large-navigate-v0 \
        --seeds "0 1 2" \
        --run-group qdrift_abl_temp \
        -- --agent.drift_temperature=${TEMP}
done
```

---

## 5. Ablation — BC pretraining budget  (`bc_pretrain_steps`)

Tests how much BC pretraining is needed before the RL phase.
Setting to 0 skips Phase 1 entirely — the drifting field has no warm attraction
target, so V_attract starts from a random bc_actor.

| Value | Description |
|-------|-------------|
| `0` | No BC pretraining (cold start) |
| `50000` | Light pretraining |
| `100000` | Half default |
| `200000` | **Default** |
| `500000` | Extended pretraining |

```bash
for BC_STEPS in 0 50000 100000 200000 500000; do
    bash scripts/submit_qdrift.sh \
        --env antmaze-large-navigate-v0 \
        --seeds "0 1 2" \
        --run-group qdrift_abl_bc \
        -- --bc_pretrain_steps=${BC_STEPS}
done
```

---

## 6. Ablation — Pessimism coefficient  (`rho`, ρ)

Controls how conservative the critic is in out-of-distribution regions.
ρ=0 → no pessimism (standard TD); ρ=1 → use min of ensemble.

| Value | Description |
|-------|-------------|
| `0.0` | No pessimism |
| `0.25` | Light pessimism |
| `0.5` | **Default** |
| `0.75` | Conservative |
| `1.0` | Fully pessimistic (equivalent to lower-bound ensemble) |

```bash
for RHO in 0.0 0.25 0.5 0.75 1.0; do
    bash scripts/submit_qdrift.sh \
        --env antmaze-large-navigate-v0 \
        --seeds "0 1 2" \
        --run-group qdrift_abl_rho \
        -- --agent.rho=${RHO}
done
```

---

## 7. Ablation — Critic ensemble size  (`num_qs`)

Tests how many Q-networks are needed for stable pessimistic TD.

| Value | Description |
|-------|-------------|
| `2` | Minimum (standard TD3-style; used by `--small`) |
| `5` | Reduced ensemble |
| `10` | **Default** |
| `20` | Large ensemble |

```bash
for NQS in 2 5 10 20; do
    bash scripts/submit_qdrift.sh \
        --env antmaze-large-navigate-v0 \
        --seeds "0 1 2" \
        --run-group qdrift_abl_nqs \
        -- --agent.num_qs=${NQS}
done
```

---

## 8. Ablation — Target network vs. live network for Q-gradient

Tests whether using the target critic (frozen EMA) for the Q-gradient is
necessary for stability, or whether the live critic works equally well.

```bash
# Default: use target critic (stable)
bash scripts/submit_qdrift.sh \
    --env antmaze-large-navigate-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_abl_targetgrad \
    -- --agent.use_target_grad=True

# Ablation: use live critic (potentially less stable but faster feedback)
bash scripts/submit_qdrift.sh \
    --env antmaze-large-navigate-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_abl_targetgrad \
    -- --agent.use_target_grad=False
```

---

## 9. Ablation — Best-of-N inference resampling  (`best_of_n`)

Tests whether drawing multiple noise samples at inference time and picking the
one with the highest Q-value improves task performance, and at what inference
cost.

| Value | Description |
|-------|-------------|
| `1` | **Default** — single forward pass |
| `4` | 4 candidates, pick highest-Q |
| `16` | 16 candidates |

```bash
for N in 1 4 16; do
    bash scripts/submit_qdrift.sh \
        --env antmaze-large-navigate-v0 \
        --seeds "0 1 2" \
        --run-group qdrift_abl_bon \
        -- --agent.best_of_n=${N}
done
```

---

## 10. Combined sweep — q_drift_scale × environment

Cross-environment validation that the best `q_drift_scale` found on
`antmaze-large` transfers to other domains.

```bash
for ENV in antmaze-medium-navigate-v0 antmaze-large-navigate-v0 antmaze-medium-stitch-v0; do
    for SCALE in 0.05 0.1 0.3; do
        bash scripts/submit_qdrift.sh \
            --env ${ENV} \
            --seeds "0 1 2" \
            --run-group qdrift_sweep_cross \
            -- --agent.q_drift_scale=${SCALE}
    done
done
```

---

## 11. Sparse reward variant

Converts dense rewards to sparse (-1 / 0) to test whether the algorithm
requires reward shaping or works with raw goal-reaching signals.

```bash
bash scripts/submit_qdrift.sh \
    --env antmaze-medium-navigate-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_sparse \
    -- --sparse=True

bash scripts/submit_qdrift.sh \
    --env antmaze-large-navigate-v0 \
    --seeds "0 1 2" \
    --run-group qdrift_sparse \
    -- --sparse=True
```

---

## Reference: default hyperparameters

| Parameter | Default | Flag |
|-----------|---------|------|
| `q_drift_scale` | `0.1` | `--agent.q_drift_scale` |
| `num_drift_samples` | `8` | `--agent.num_drift_samples` |
| `drift_temperature` | `1.0` | `--agent.drift_temperature` |
| `num_qs` | `10` | `--agent.num_qs` |
| `rho` | `0.5` | `--agent.rho` |
| `discount` | `0.99` | `--agent.discount` |
| `tau` | `0.005` | `--agent.tau` |
| `best_of_n` | `1` | `--agent.best_of_n` |
| `use_target_grad` | `True` | `--agent.use_target_grad` |
| `lr` | `3e-4` | `--agent.lr` |
| `batch_size` | `256` | `--agent.batch_size` |
| `bc_pretrain_steps` | `200000` | `--bc_pretrain_steps` |
| `offline_steps` | `1000000` | `--offline_steps` |
