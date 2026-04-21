"""Logging utilities for Q-Drifting experiments.

Output layout per experiment:
  exp/<run_group>/<env>/<hash>/
  ├── flags.json            — all CLI flags + agent hyperparameters
  ├── experiment_info.json  — system metadata, dataset shape, param count, git hash
  ├── bc_agent.csv          — per-step metrics during BC pretraining
  ├── offline_agent.csv     — per-step metrics during offline RL
  ├── eval.csv              — per-evaluation-checkpoint metrics (mean ± std, min, max)
  ├── results_summary.json  — best/final eval score, total wall time, steps/sec
  └── token.tk              — written on clean completion (used for deduplication)

Every CSV row carries:
  step       — global training step
  wall_time  — seconds elapsed since the Python process started

Use pandas to load for analysis:
  df = pd.read_csv('eval.csv')
  plt.plot(df['step'], df['success'], label='mean')
  plt.fill_between(df['step'],
                   df['success'] - df['success_std'],
                   df['success'] + df['success_std'], alpha=0.2)
"""
import os
import csv
import json
import time
import shutil
import hashlib
import platform
import socket
import subprocess

import absl.flags as flags
import ml_collections

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string, e.g. '2h 3m 14s'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h > 0:
        parts.append(f'{h}h')
    if m > 0 or h > 0:
        parts.append(f'{m}m')
    parts.append(f'{s}s')
    return ' '.join(parts)


def _get_git_hash() -> str:
    """Return the current git commit hash (short), or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def get_csv_header(path: str):
    with open(path, 'r', newline='') as f:
        return csv.DictReader(f).fieldnames


# ---------------------------------------------------------------------------
# CsvLogger
# ---------------------------------------------------------------------------

class CsvLogger:
    """Streams metrics to a CSV file.

    - 'step' and 'wall_time' are prepended to every row automatically.
    - New keys that appear after the header is written are appended to the
      header (earlier rows will have empty cells for those keys).
    - The file is flushed after every row so partial logs survive crashes.
    """

    def __init__(self, path: str, start_time: float = None):
        self.path = path
        self.header = None
        self.file = None
        # Allow a shared start_time so wall_time is consistent across all
        # loggers in the same experiment.
        self._start_time = start_time if start_time is not None else time.time()

    def log(self, row: dict, step: int):
        """Write one row of metrics.

        Args:
            row:  Dict of metric name -> scalar value. JAX/numpy scalars are
                  converted to Python floats automatically.
            step: The global training step this row corresponds to.
        """
        cleaned = {}
        for k, v in row.items():
            if hasattr(v, 'item'):
                try:
                    cleaned[k] = v.item()
                except (ValueError, TypeError):
                    pass  # skip non-scalar arrays
            elif isinstance(v, (int, float, str, bool)):
                cleaned[k] = v

        # Prepend step and wall_time so they are always the first two columns.
        ordered = {
            'step': step,
            'wall_time': round(time.time() - self._start_time, 2),
        }
        ordered.update(cleaned)

        if self.file is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.file = open(self.path, 'w', newline='')
            if self.header is None:
                self.header = list(ordered.keys())
                self.file.write(','.join(self.header) + '\n')
            self.file.write(','.join(str(ordered.get(k, '')) for k in self.header) + '\n')
        else:
            for k in ordered:
                if k not in self.header:
                    self.header.append(k)
            self.file.write(','.join(str(ordered.get(k, '')) for k in self.header) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def save(self, dst_path: str):
        """Flush and copy to a backup path (used for checkpointing)."""
        if self.file is not None:
            self.file.flush()
        if os.path.exists(self.path):
            shutil.copyfile(self.path, dst_path)

    def restore(self, src_path: str):
        """Restore from a backup path and resume appending."""
        shutil.copyfile(src_path, self.path)
        self.header = get_csv_header(self.path)
        self.file = open(self.path, 'a', newline='')


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """Central logger for one experiment run.

    Responsibilities:
      - Creates and manages per-phase CsvLoggers (bc_agent, offline_agent, eval).
      - Tracks wall-time, steps/sec, and the best eval score seen so far.
      - Writes experiment_info.json at startup (static metadata).
      - Writes results_summary.json at completion (performance results).

    Usage in main.py::

        logger = ExperimentLogger(save_dir, phases=['bc_agent', 'offline_agent', 'eval'])
        logger.save_experiment_info(FLAGS, train_dataset, agent, param_count)

        for step in range(...):
            agent, info = agent.update(batch)
            if step % log_interval == 0:
                logger.log(info, 'offline_agent', step=step)
            if step % eval_interval == 0:
                eval_info = evaluate(...)
                logger.log(eval_info, 'eval', step=step)

        logger.save_results_summary(bc_steps, offline_steps)
        logger.close()
    """

    def __init__(self, save_dir: str, phases: list, eval_metric: str = 'success'):
        """
        Args:
            save_dir:    Directory where all output files are written.
            phases:      List of CSV logger names, e.g. ['bc_agent', 'offline_agent', 'eval'].
            eval_metric: The key in eval rows used to track best performance.
                         Typically 'success' for navigation tasks or 'episode_return'
                         for continuous control.
        """
        self.save_dir = save_dir
        self.eval_metric = eval_metric
        self._start_time = time.time()

        self.csv_loggers = {
            phase: CsvLogger(
                os.path.join(save_dir, f'{phase}.csv'),
                start_time=self._start_time,
            )
            for phase in phases
        }

        # Best eval tracking
        self._best_eval: float = None
        self._best_eval_step: int = None
        self._best_eval_std: float = None
        self._last_eval: float = None
        self._last_eval_step: int = None
        self._last_eval_std: float = None

        # Steps-per-second tracking
        self._total_steps = 0

    # ------------------------------------------------------------------
    # Core logging interface
    # ------------------------------------------------------------------

    def log(self, data: dict, phase: str, step: int):
        """Log a metric dict to the given phase's CSV.

        Also updates best/last eval tracking when phase == 'eval'.
        """
        assert phase in self.csv_loggers, (
            f"Unknown phase '{phase}'. Known phases: {list(self.csv_loggers)}"
        )
        self.csv_loggers[phase].log(data, step=step)
        self._total_steps = max(self._total_steps, step)

        if phase == 'eval':
            val = data.get(self.eval_metric)
            std = data.get(f'{self.eval_metric}_std')
            if val is not None:
                val = float(val) if hasattr(val, '__float__') else val
                std = float(std) if std is not None else None
                self._last_eval = val
                self._last_eval_step = step
                self._last_eval_std = std
                if self._best_eval is None or val > self._best_eval:
                    self._best_eval = val
                    self._best_eval_step = step
                    self._best_eval_std = std

    # ------------------------------------------------------------------
    # Console helpers
    # ------------------------------------------------------------------

    def elapsed(self) -> float:
        return time.time() - self._start_time

    def steps_per_sec(self, steps_done: int) -> float:
        e = self.elapsed()
        return steps_done / e if e > 0 else 0.0

    def format_eval_line(self, eval_info: dict) -> str:
        """Return a compact formatted string of key eval metrics for console."""
        parts = []
        for k, v in sorted(eval_info.items()):
            if '_std' in k or '_min' in k or '_max' in k or '_median' in k:
                continue  # shown inline with their base key
            if not isinstance(v, (int, float)):
                continue
            std = eval_info.get(f'{k}_std')
            if std is not None:
                parts.append(f'{k}={v:.3f}±{std:.3f}')
            else:
                parts.append(f'{k}={v:.3f}')
        return '  '.join(parts)

    # ------------------------------------------------------------------
    # Checkpointing helpers
    # ------------------------------------------------------------------

    def save_all(self):
        """Flush all CSV loggers to backup files."""
        for phase, logger in self.csv_loggers.items():
            logger.save(os.path.join(self.save_dir, f'{phase}_sv.csv'))

    def restore_all(self):
        """Restore all CSV loggers from backup files (called on resume)."""
        for phase, logger in self.csv_loggers.items():
            bak = os.path.join(self.save_dir, f'{phase}_sv.csv')
            if os.path.exists(bak):
                logger.restore(bak)

    def close(self):
        for logger in self.csv_loggers.values():
            logger.close()

    # ------------------------------------------------------------------
    # JSON metadata files
    # ------------------------------------------------------------------

    def save_experiment_info(
        self,
        flags_obj,
        train_dataset,
        agent,
        param_count: int,
    ):
        """Write experiment_info.json with static metadata about this run.

        This is meant to be called once, right before training begins.
        It records everything needed to reproduce or understand the run:
        hyperparameters, system info, dataset dimensions, and network size.

        Args:
            flags_obj:   The absl FLAGS object.
            train_dataset: The Dataset used for training (for shape info).
            agent:       The initialized agent (for module names).
            param_count: Total non-target param count (computed in main.py).
        """
        import jax

        agent_cfg = dict(flags_obj.agent) if hasattr(flags_obj.agent, 'to_dict') else {}

        info = {
            'timestamp_start': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'git_hash': _get_git_hash(),
            'env_name': flags_obj.env_name,
            'run_group': flags_obj.run_group,
            'seed': flags_obj.seed,
            'save_dir': flags_obj.save_dir,
            'phases': {
                'bc_pretrain_steps': flags_obj.bc_pretrain_steps,
                'offline_steps': flags_obj.offline_steps,
            },
            'platform': {
                'hostname': socket.gethostname(),
                'os': platform.system(),
                'python': platform.python_version(),
                'jax_version': jax.__version__,
                'jax_devices': [str(d) for d in jax.devices()],
            },
            'dataset': {
                'size': int(train_dataset.size),
                'observation_dim': int(train_dataset['observations'].shape[-1]),
                'action_dim': int(train_dataset['actions'].shape[-1]),
                'num_episodes': int(len(train_dataset.terminal_locs)),
            },
            'network': {
                'param_count': param_count,
                'modules': [
                    k.replace('modules_', '')
                    for k in agent.network.params.keys()
                ],
            },
            'hyperparameters': {
                k: (v if not isinstance(v, (list, tuple)) else list(v))
                for k, v in agent_cfg.items()
                if k not in ('ob_dims', 'action_dim', 'horizon_length')
            },
        }

        path = os.path.join(self.save_dir, 'experiment_info.json')
        with open(path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f'Experiment info saved to {path}')

    def save_results_summary(
        self,
        bc_steps: int,
        offline_steps: int,
        inference: dict = None,
    ) -> dict:
        """Write results_summary.json and return it as a dict.

        Called once at the very end of training, after the inference benchmark
        has already been logged to benchmark.csv via logger.log().

        Args:
            bc_steps:      BC pretraining steps completed.
            offline_steps: Offline RL steps completed.
            inference:     Optional dict of headline inference numbers to embed
                           in the summary (single_obs_latency_ms, kstep_speedups,
                           etc.) — avoids having to open inference_benchmark.json
                           for a quick result check.
        """
        elapsed = self.elapsed()
        total_steps = bc_steps + offline_steps

        summary = {
            'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'total_wall_time_sec': round(elapsed, 1),
            'total_wall_time_human': _fmt_duration(elapsed),
            'bc_pretrain_steps': bc_steps,
            'offline_steps': offline_steps,
            'total_steps': total_steps,
            'avg_steps_per_sec': round(total_steps / elapsed, 1) if elapsed > 0 else None,
            'eval': {
                'metric': self.eval_metric,
                'best': {
                    'value': self._best_eval,
                    'std': self._best_eval_std,
                    'step': self._best_eval_step,
                },
                'final': {
                    'value': self._last_eval,
                    'std': self._last_eval_std,
                    'step': self._last_eval_step,
                },
            },
        }

        if inference is not None:
            summary['inference'] = inference

        path = os.path.join(self.save_dir, 'results_summary.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary


# ---------------------------------------------------------------------------
# Flag utilities (unchanged from original)
# ---------------------------------------------------------------------------

def get_hash(s: str) -> str:
    encoded = s.encode('utf-8')
    h = hashlib.sha256()
    h.update(encoded)
    return h.hexdigest()[:16]


def get_exp_name(flags_obj) -> str:
    """Return a short deterministic experiment name derived from all flag values.

    The hash changes whenever any flag changes, so each unique hyperparameter
    combination gets its own directory automatically.
    """
    return get_hash(flags_obj.flags_into_string())


def get_flag_dict() -> dict:
    """Return all CLI flags as a plain Python dict (suitable for json.dump)."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict
