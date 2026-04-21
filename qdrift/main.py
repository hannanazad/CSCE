"""Training entry point for Q-Drifting (offline only).

Two-phase training
------------------
Phase 1 — BC Pretraining (bc_pretrain_steps)
    Train `bc_actor` via behaviour cloning (MSE loss on dataset actions).
    The bc_actor learns a smooth, generalized mapping obs -> action that
    captures the behavioral distribution.  It is frozen after this phase.

Phase 2 — Offline RL (offline_steps)
    Train the drifting actor and critic.  The frozen bc_actor provides the
    attraction target in the drifting field (V_attract = bc_actor(obs) - x_gen),
    giving a stable, state-conditioned behavioural prior.  The Q-gradient
    term then steers away from that prior toward higher-value regions.

Output files (written to exp/<run_group>/<env>/<hash>/)
-------------------------------------------------------
  flags.json           — all CLI flags + agent config
  experiment_info.json — system metadata, dataset shape, param count, git hash
  bc_agent.csv         — BC pretraining metrics (step, wall_time, bc/loss, ...)
  offline_agent.csv    — offline RL metrics   (step, wall_time, critic/*, actor/*)
  eval.csv             — evaluation results   (step, wall_time, success±std, ...)
  results_summary.json — best/final eval score, total wall time, steps/sec
  token.tk             — written on clean completion (for job deduplication)
"""
import glob
import json
import os
import random
import time

import jax
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from evaluation import evaluate
from log_utils import ExperimentLogger, get_exp_name, get_flag_dict
from utils.datasets import Dataset
from utils.flax_utils import save_agent, restore_agent
from utils.inference_benchmark import benchmark_inference, print_benchmark_summary

# ---------------------------------------------------------------------------
# Flag definitions
# ---------------------------------------------------------------------------

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group (appears in save_dir path).')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'OGBench environment name.')
flags.DEFINE_string('save_dir', 'exp/', 'Root directory for logs and checkpoints.')

# Training budget
flags.DEFINE_integer('bc_pretrain_steps', 200000,
                     'BC pretraining steps (Phase 1). Set to 0 to skip.')
flags.DEFINE_integer('offline_steps', 1000000,
                     'Offline RL training steps (Phase 2).')

# Logging / evaluation cadence
flags.DEFINE_integer('log_interval', 5000,  'Write to CSV every N steps.')
flags.DEFINE_integer('eval_interval', 50000, 'Run evaluation every N steps.')
flags.DEFINE_integer('save_interval', 50000, 'Save checkpoint every N steps (0 = never).')
flags.DEFINE_integer('eval_episodes', 50, 'Episodes per evaluation.')
flags.DEFINE_integer('video_episodes', 0, 'Extra video-render episodes per eval.')
flags.DEFINE_integer('video_frame_skip', 3, 'Render every N steps for video.')

# Agent config
config_flags.DEFINE_config_file('agent', 'agents/qdrift.py', lock_config=False)

# Dataset
flags.DEFINE_float('dataset_proportion', 1.0,
                   'Fraction of the dataset to use (useful for ablations).')
flags.DEFINE_integer('dataset_replace_interval', 1000,
                     'Rotate to the next dataset chunk every N steps (chunked datasets only).')
flags.DEFINE_string('ogbench_dataset_dir',
                    os.environ.get('OGBENCH_DATA_DIR', None),
                    'Pre-downloaded OGBench dataset directory. '
                    'Defaults to $OGBENCH_DATA_DIR. '
                    'Omit to let OGBench download data automatically.')

# Task options
flags.DEFINE_integer('horizon_length', 1,
                     'N-step return horizon. Keep at 1 (no action chunking).')
flags.DEFINE_bool('sparse', False,
                  'Convert dense rewards to sparse (-1/0) before training.')

# Housekeeping
flags.DEFINE_bool('auto_cleanup', True,
                  'Delete intermediate checkpoint files on clean completion.')
flags.DEFINE_bool('debug', False,
                  'Debug mode: very short run (100+100 steps, 2 eval episodes).')
flags.DEFINE_bool('small', False,
                  'CPU-friendly preset: smaller networks, fewer ensemble members, '
                  'reduced batch size and step counts. Runs in ~3-6 hours on CPU '
                  'vs ~55-110 hours for the default config. '
                  'Equivalent to: --agent.num_qs=2 --agent.actor_hidden_dims=256,256,256 '
                  '--agent.value_hidden_dims=256,256,256 --agent.bc_hidden_dims=256,256,256 '
                  '--agent.batch_size=128 --agent.num_drift_samples=4 '
                  '--bc_pretrain_steps=100000 --offline_steps=300000')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _process_dataset(ds, flags_obj):
    """Convert a raw dataset dict into a Dataset and apply any flag-level transforms."""
    ds = Dataset.create(**ds)
    if flags_obj.dataset_proportion < 1.0:
        n = int(len(ds['masks']) * flags_obj.dataset_proportion)
        ds = Dataset.create(**{k: v[:n] for k, v in ds.items()})
    if flags_obj.sparse:
        rewards = (ds['rewards'] != 0.0) * -1.0
        ds = Dataset.create(**dict(ds, rewards=rewards))
    return ds


def _load_env_and_dataset(flags_obj):
    """Load environment + dataset according to the dataset flags.

    Returns:
        env, eval_env, train_dataset (Dataset), dataset_paths (list|None).
        dataset_paths is None for single-file datasets (no rotation needed).
    """
    if flags_obj.ogbench_dataset_dir is not None:
        matching = [
            f for f in sorted(glob.glob(f'{flags_obj.ogbench_dataset_dir}/*.npz'))
            if '-val.npz' not in f and flags_obj.env_name in os.path.basename(f)
        ]
        if len(matching) <= 1:
            print(f'Single-file dataset: {flags_obj.ogbench_dataset_dir}')
            env, eval_env, raw_train, _ = make_env_and_datasets(
                flags_obj.env_name, dataset_dir=flags_obj.ogbench_dataset_dir
            )
            return env, eval_env, _process_dataset(raw_train, flags_obj), None
        else:
            if flags_obj.dataset_proportion < 1.0:
                n = max(1, int(len(matching) * flags_obj.dataset_proportion))
                print(f'Using {n}/{len(matching)} dataset chunks')
                matching = matching[:n]
            print(f'Chunked dataset: {len(matching)} files')
            env, eval_env, raw_train, _ = make_ogbench_env_and_datasets(
                flags_obj.env_name,
                dataset_path=matching[0],
                compact_dataset=False,
            )
            return env, eval_env, _process_dataset(raw_train, flags_obj), matching
    else:
        env, eval_env, raw_train, _ = make_env_and_datasets(flags_obj.env_name)
        return env, eval_env, _process_dataset(raw_train, flags_obj), None


def _sample_batch(dataset, config, horizon_length, discount):
    return dataset.sample_sequence(
        config['batch_size'],
        sequence_length=horizon_length,
        discount=discount,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    # ---- Debug overrides ------------------------------------------------
    if FLAGS.debug:
        FLAGS.bc_pretrain_steps = 100
        FLAGS.offline_steps = 100
        FLAGS.eval_interval = 100
        FLAGS.log_interval = 50
        FLAGS.save_interval = 0
        FLAGS.eval_episodes = 2

    # ---- Small (CPU-friendly) preset ------------------------------------
    # The default config (num_qs=10, 4×512 MLPs, K=8, 1.2M steps) requires
    # ~16 GFLOPs per training step and takes ~55-110 hours on CPU.
    # This preset reduces that to ~1.3 GFLOPs/step and ~3-6 hours on CPU
    # by shrinking the three main cost drivers:
    #   1. Critic ensemble: 10 → 2 members  (biggest win, ~5× cheaper)
    #   2. Hidden dims: 4×512 → 3×256       (~8× fewer params per network)
    #   3. Drift samples K: 8 → 4           (2× cheaper actor loss)
    #   4. Batch size: 256 → 128            (2× cheaper per step)
    #   5. Step budget: 1.2M → 400K         (3× fewer steps)
    # Performance will be lower than GPU results but still meaningful for
    # ablations, debugging, and initial hyperparameter exploration.
    if FLAGS.small:
        FLAGS.bc_pretrain_steps = min(FLAGS.bc_pretrain_steps, 100000)
        FLAGS.offline_steps     = min(FLAGS.offline_steps,     300000)
        FLAGS.eval_interval     = min(FLAGS.eval_interval,     30000)
        FLAGS.log_interval      = min(FLAGS.log_interval,      3000)
        FLAGS.agent.num_qs              = 2
        FLAGS.agent.batch_size          = 128
        FLAGS.agent.num_drift_samples   = 4
        FLAGS.agent.actor_hidden_dims   = (256, 256, 256)
        FLAGS.agent.value_hidden_dims   = (256, 256, 256)
        FLAGS.agent.bc_hidden_dims      = (256, 256, 256)
        print('[--small] CPU-friendly preset applied.')
        print('  num_qs=2, hidden=3×256, batch=128, K=4, '
              f'bc={FLAGS.bc_pretrain_steps:,} + offline={FLAGS.offline_steps:,} steps')

    # ---- Save directory -------------------------------------------------
    exp_name = get_exp_name(FLAGS)
    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, FLAGS.run_group, FLAGS.env_name, exp_name
    )
    print(f'\nExperiment : {FLAGS.env_name}  seed={FLAGS.seed}')
    print(f'Save dir   : {FLAGS.save_dir}\n')

    # ---- Data loading ---------------------------------------------------
    env, eval_env, train_dataset, dataset_paths = _load_env_and_dataset(FLAGS)
    dataset_idx = 0

    # ---- Seeding --------------------------------------------------------
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # ---- Agent construction ---------------------------------------------
    config = FLAGS.agent
    config['horizon_length'] = FLAGS.horizon_length
    example_batch = train_dataset.sample(())

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Param count (exclude target networks — they are copies, not extra params)
    trainable_params = {k: v for k, v in agent.network.params.items()
                        if 'target' not in k}
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(trainable_params))
    print(f'Network modules : {list(trainable_params.keys())}')
    print(f'Param count     : {param_count:,}')
    _print_hardware_and_eta(agent, train_dataset, config, FLAGS)

    # ---- Logger setup ---------------------------------------------------
    phases = ['eval']
    if FLAGS.bc_pretrain_steps > 0:
        phases.insert(0, 'bc_agent')
    if FLAGS.offline_steps > 0:
        phases.append('offline_agent')
    phases.append('benchmark')  # inference timing, written once at end of training

    logger = ExperimentLogger(
        save_dir=FLAGS.save_dir,
        phases=phases,
        # 'success' for navigation tasks; falls back gracefully if absent
        eval_metric='success',
    )

    # ---- Resume detection -----------------------------------------------
    load_stage = None
    load_step = None

    if os.path.isdir(FLAGS.save_dir):
        print(f'Detected existing run at {FLAGS.save_dir}')
        if os.path.exists(os.path.join(FLAGS.save_dir, 'token.tk')):
            print('Run already completed. Exiting.')
            return
        try:
            with open(os.path.join(FLAGS.save_dir, 'progress.tk')) as f:
                load_stage, load_step = f.read().split(',')
            load_step = int(load_step)
            agent = restore_agent(agent, FLAGS.save_dir, load_step)
            logger.restore_all()
            assert load_stage in ('bc', 'offline'), f'Unknown stage: {load_stage}'
            print(f'Resumed from {load_stage} step {load_step}')
        except Exception:
            load_stage = None
            load_step = None
            print('Could not restore; starting fresh.')
    else:
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
            json.dump(get_flag_dict(), f, indent=2)
        logger.save_experiment_info(FLAGS, train_dataset, agent, param_count)

    horizon_length = FLAGS.horizon_length
    discount = config['discount']
    action_dim = example_batch['actions'].shape[-1]

    # ---- Dataset rotation helper ----------------------------------------
    def maybe_rotate(step):
        """Rotate to next dataset chunk if the interval has elapsed."""
        nonlocal train_dataset, dataset_idx
        if (dataset_paths is not None
                and FLAGS.dataset_replace_interval != 0
                and step % FLAGS.dataset_replace_interval == 0):
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            raw, _ = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = _process_dataset(raw, FLAGS)
            print(f'  [step {step}] rotated to dataset chunk {dataset_idx}')

    # ---- Checkpoint helper ----------------------------------------------
    def checkpoint(stage: str, step: int):
        if FLAGS.save_interval > 0 and step % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, step)
            logger.save_all()
            with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'w') as f:
                f.write(f'{stage},{step}')

    # ---- Eval helper ----------------------------------------------------
    def run_eval(log_step: int, phase_label: str):
        eval_info, _, _ = evaluate(
            agent=agent,
            env=eval_env,
            action_dim=action_dim,
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
        )
        logger.log(eval_info, 'eval', step=log_step)
        summary = logger.format_eval_line(eval_info)
        elapsed = logger.elapsed()
        sps = logger.steps_per_sec(log_step)
        print(
            f'[{phase_label} step {log_step:>8,}]'
            f'  {summary}'
            f'  |  {sps:.0f} sps  {_fmt_elapsed(elapsed)}'
        )
        return eval_info

    # ====================================================================
    # Phase 1: BC Pretraining
    # ====================================================================

    if load_stage == 'offline':
        bc_start = FLAGS.bc_pretrain_steps + 1  # already completed
    elif load_stage == 'bc' and load_step is not None:
        bc_start = load_step + 1
    else:
        bc_start = 1

    if FLAGS.bc_pretrain_steps > 0:
        print(f'\n{"="*60}')
        print(f'Phase 1: BC Pretraining ({FLAGS.bc_pretrain_steps:,} steps)')
        print(f'{"="*60}')

    pbar = tqdm.trange(
        bc_start, FLAGS.bc_pretrain_steps + 1,
        desc='BC Pretrain', unit='step', leave=True,
    )
    for i in pbar:
        maybe_rotate(i)
        batch = _sample_batch(train_dataset, config, horizon_length, discount)
        agent, info = agent.bc_update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(info, 'bc_agent', step=i)
            pbar.set_postfix({
                'loss': f'{float(info.get("bc/loss", float("nan"))):.4f}',
                'err':  f'{float(info.get("bc/action_err", float("nan"))):.4f}',
                'sps':  f'{logger.steps_per_sec(i):.0f}',
            })

        checkpoint('bc', i)

    if FLAGS.bc_pretrain_steps > 0:
        print('BC pretraining complete.\n')

    # ====================================================================
    # Phase 2: Offline RL
    # ====================================================================

    if load_stage == 'offline' and load_step is not None:
        offline_start = load_step + 1
    else:
        offline_start = 1

    if FLAGS.offline_steps > 0:
        print(f'{"="*60}')
        print(f'Phase 2: Offline RL ({FLAGS.offline_steps:,} steps)')
        print(f'{"="*60}')

    pbar = tqdm.trange(
        offline_start, FLAGS.offline_steps + 1,
        desc='Offline RL', unit='step', leave=True,
    )
    for i in pbar:
        log_step = FLAGS.bc_pretrain_steps + i
        maybe_rotate(i)

        batch = _sample_batch(train_dataset, config, horizon_length, discount)
        agent, info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(info, 'offline_agent', step=log_step)
            pbar.set_postfix({
                'q':      f'{float(info.get("critic/q_mean",      float("nan"))):.2f}',
                'td_err': f'{float(info.get("critic/td_error_mean", float("nan"))):.4f}',
                'dloss':  f'{float(info.get("actor/drift_loss",   float("nan"))):.4f}',
                'att':    f'{float(info.get("actor/v_attract_mag", float("nan"))):.3f}',
                'rep':    f'{float(info.get("actor/v_repulse_mag", float("nan"))):.3f}',
                'qmag':   f'{float(info.get("actor/v_q_mag",      float("nan"))):.3f}',
                'sps':    f'{logger.steps_per_sec(log_step):.0f}',
            })

        if i == FLAGS.offline_steps or (
            FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0
        ):
            run_eval(log_step, phase_label='Offline')

        checkpoint('offline', i)

    # ====================================================================
    # Inference benchmark  (core research claim: 1-step is fast)
    # Runs before logger.close() so results flow through the CSV system.
    # ====================================================================
    total_steps = FLAGS.bc_pretrain_steps + FLAGS.offline_steps

    print(f'\n{"="*60}')
    print('Running inference benchmark ...')
    bench_results = benchmark_inference(
        agent=agent,
        example_obs=example_batch['observations'],
        action_dim=action_dim,
        n_warmup=10 if FLAGS.debug else 50,
        n_trials=20 if FLAGS.debug else 200,
    )
    print_benchmark_summary(bench_results)

    # Flatten benchmark results into a single dict and log to benchmark.csv.
    # step = total training steps so the row aligns with the end of training.
    bench_row = _flatten_benchmark(bench_results)
    logger.log(bench_row, 'benchmark', step=total_steps)

    # Also save the full nested structure as inference_benchmark.json for
    # convenient programmatic access (e.g. in analysis notebooks).
    bench_path = os.path.join(FLAGS.save_dir, 'inference_benchmark.json')
    with open(bench_path, 'w') as f:
        json.dump(bench_results, f, indent=2)
    print(f'Inference benchmark saved to {bench_path}')

    # ====================================================================
    # Finalize — close loggers, write summary, mark complete
    # ====================================================================

    logger.close()

    so = bench_results['single_obs']
    bench_summary = {
        'single_obs_latency_ms':      so['latency_ms_mean'],
        'single_obs_latency_std_ms':  so['latency_ms_std'],
        'single_obs_throughput_per_sec': so['throughput_per_sec'],
        'kstep_speedups': {
            k: v['qdrift_speedup']
            for k, v in bench_results['kstep_comparison'].items()
        },
    }

    summary = logger.save_results_summary(
        bc_steps=FLAGS.bc_pretrain_steps,
        offline_steps=FLAGS.offline_steps,
        inference=bench_summary,
    )

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write('done')

    # Remove intermediate checkpoint files
    if FLAGS.auto_cleanup:
        for fname in os.listdir(FLAGS.save_dir):
            fpath = os.path.join(FLAGS.save_dir, fname)
            if os.path.isfile(fpath) and fname.startswith('params'):
                os.remove(fpath)

    # Print final summary
    ev = summary['eval']
    best = ev['best']
    final = ev['final']
    inf = summary.get('inference', {})
    print(f'\n{"="*60}')
    print(f'Run complete  —  {summary["total_wall_time_human"]}'
          f'  ({summary["avg_steps_per_sec"]:.0f} training sps)')
    print(f'  Best  {ev["metric"]}: {best["value"]:.4f}'
          + (f' ± {best["std"]:.4f}' if best["std"] is not None else '')
          + f'  @ step {best["step"]:,}')
    print(f'  Final {ev["metric"]}: {final["value"]:.4f}'
          + (f' ± {final["std"]:.4f}' if final["std"] is not None else '')
          + f'  @ step {final["step"]:,}')
    if inf:
        lat = inf["single_obs_latency_ms"]
        thr = inf["single_obs_throughput_per_sec"]
        sp8  = inf["kstep_speedups"].get("8",  "?")
        sp16 = inf["kstep_speedups"].get("16", "?")
        sp32 = inf["kstep_speedups"].get("32", "?")
        print(f'  Inference  : {lat:.3f} ms / action  ({thr:.0f} inf/sec)')
        print(f'  Speedup vs 8-step : {sp8:.1f}×  |  16-step : {sp16:.1f}×  |  32-step : {sp32:.1f}×')
    print(f'Results saved to: {FLAGS.save_dir}')
    print(f'{"="*60}\n')


def _print_hardware_and_eta(agent, train_dataset, config, flags_obj):
    """Time a few real training steps and print a concrete ETA for the full run.

    This runs before any logging or checkpointing so the user knows upfront
    how long to expect the run to take on their hardware, and whether to use
    --small for CPU execution.

    The estimate covers one update() call (offline RL step), which is the
    more expensive of the two phases.  BC pretraining is cheaper per step.
    """
    import time as _time

    N_WARMUP = 5   # trigger JIT compilation (these are discarded)
    N_TIMED  = 10  # steps to average over for the ETA

    horizon  = flags_obj.horizon_length
    discount = config['discount']

    # Warmup: first call compiles the JAX graph; subsequent ones hit cache.
    _agent = agent
    for _ in range(N_WARMUP):
        batch = train_dataset.sample_sequence(
            config['batch_size'], sequence_length=horizon, discount=discount
        )
        _agent, _ = _agent.update(batch)

    # Timed steps
    t0 = _time.perf_counter()
    for _ in range(N_TIMED):
        batch = train_dataset.sample_sequence(
            config['batch_size'], sequence_length=horizon, discount=discount
        )
        _agent, _ = _agent.update(batch)
    # block_until_ready ensures async XLA dispatch completes before we stop the clock
    jax.block_until_ready(jax.tree_util.tree_leaves(_agent.network.params))
    elapsed = _time.perf_counter() - t0

    sec_per_step = elapsed / N_TIMED
    total_steps  = flags_obj.bc_pretrain_steps + flags_obj.offline_steps
    eta_sec      = sec_per_step * total_steps

    device = str(jax.devices()[0])
    on_cpu = 'cpu' in device.lower()

    print(f'\nHardware       : {device}')
    print(f'Step time      : {sec_per_step*1000:.1f} ms/step  '
          f'({1/sec_per_step:.0f} steps/sec)')
    print(f'Total steps    : {total_steps:,}  '
          f'(BC {flags_obj.bc_pretrain_steps:,} + offline {flags_obj.offline_steps:,})')
    print(f'Estimated time : {_fmt_elapsed(eta_sec)}  ({eta_sec/3600:.1f} h)')

    if on_cpu and eta_sec > 6 * 3600 and not flags_obj.small:
        print()
        print('  ⚠  Running on CPU with the default config will take '
              f'~{eta_sec/3600:.0f} hours.')
        print('     Re-run with --small for a CPU-viable preset (~3-6 h):')
        print('       python main.py --small --env_name=... --seed=0')
        print('     Or run on a GPU for full-scale results.')
    elif on_cpu and flags_obj.small:
        print('  ✓  --small preset: estimated CPU time is within reach.')
    print()


def _flatten_benchmark(bench: dict) -> dict:
    """Flatten the nested benchmark dict into a single-level dict for CSV logging.

    benchmark.csv ends up with columns like:
      single_obs_latency_ms, single_obs_throughput_per_sec,
      batch_16_throughput_per_sec, batch_256_throughput_per_sec,
      kstep_1_latency_ms, kstep_8_latency_ms, kstep_8_speedup,
      kstep_16_latency_ms, kstep_16_speedup, ...
    """
    row = {}
    so = bench['single_obs']
    row['single_obs_latency_ms']       = so['latency_ms_mean']
    row['single_obs_latency_std_ms']   = so['latency_ms_std']
    row['single_obs_throughput_per_sec'] = so['throughput_per_sec']

    for bs, entry in bench['batch_throughput'].items():
        row[f'batch_{bs}_latency_ms']       = entry['latency_ms']
        row[f'batch_{bs}_throughput_per_sec'] = entry['throughput_per_sec']

    for K, entry in bench['kstep_comparison'].items():
        row[f'kstep_{K}_latency_ms']  = entry['latency_ms_mean']
        row[f'kstep_{K}_latency_std'] = entry['latency_ms_std']
        row[f'kstep_{K}_speedup']     = entry['qdrift_speedup']

    return row


def _fmt_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f'{h}h{m:02d}m'
    if m > 0:
        return f'{m}m{s:02d}s'
    return f'{s}s'


if __name__ == '__main__':
    app.run(main)
