"""Episode evaluation for Q-Drifting.

Runs the agent in the environment for a fixed number of episodes and
returns aggregated statistics suitable for CSV logging and console display.

Returned stats dict includes, for every tracked metric:
  {key}         — mean across episodes
  {key}_std     — standard deviation (for error bars in plots)
  {key}_min     — minimum observed value
  {key}_max     — maximum observed value
  {key}_median  — robust to outlier episodes

Always-present keys:
  episode_return         — cumulative undiscounted reward per episode
  episode_length         — environment steps per episode
  inference_latency_ms   — wall-clock time for one sample_actions() call (ms)

The inference_latency_ms columns provide a live record of inference speed
alongside task performance, so learning curves and timing are aligned on the
same step axis.  Use utils/inference_benchmark.py for a more thorough
standalone benchmark that includes K-step comparisons and batch throughput.
"""
import time
from collections import defaultdict
from functools import partial

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Wrap a function so a fresh RNG key is split off before each call."""
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)
    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Recursively flatten a nested dict, joining keys with `sep`."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _aggregate(per_episode: dict) -> dict:
    """Compute mean/std/min/max/median for each per-episode metric list.

    Args:
        per_episode: Dict mapping metric name -> list of per-episode scalars.

    Returns:
        Flat dict with {key}, {key}_std, {key}_min, {key}_max, {key}_median.
    """
    out = {}
    for k, v in per_episode.items():
        arr = np.array(v, dtype=float)
        out[k] = float(np.mean(arr))
        out[f'{k}_std'] = float(np.std(arr))
        out[f'{k}_min'] = float(np.min(arr))
        out[f'{k}_max'] = float(np.max(arr))
        out[f'{k}_median'] = float(np.median(arr))
    return out


def evaluate(
    agent,
    env,
    num_eval_episodes: int = 50,
    num_video_episodes: int = 0,
    video_frame_skip: int = 3,
    action_dim: int = None,
    extra_sample_kwargs: dict = {},
):
    """Roll out the agent and collect statistics over multiple episodes.

    The agent's `sample_actions` is called with a fresh RNG key each step.
    Actions are clipped to [-1, 1] before being passed to the environment.

    Args:
        agent:               The trained agent (must have `sample_actions`).
        env:                 Gym-compatible evaluation environment.
        num_eval_episodes:   Number of episodes counted toward statistics.
        num_video_episodes:  Additional episodes rendered to video (not counted
                             in statistics). Requires env.render().
        video_frame_skip:    Render every N steps (reduces video size).
        action_dim:          Action dimension (used to reshape chunk outputs).
        extra_sample_kwargs: Extra keyword args forwarded to sample_actions.

    Returns:
        stats (dict):   Aggregated metrics — see module docstring for format.
        trajs (list):   List of trajectory dicts for eval episodes.
        renders (list): List of rendered frame arrays for video episodes.
    """
    actor_fn = supply_rng(
        partial(agent.sample_actions, **extra_sample_kwargs),
        rng=jax.random.PRNGKey(np.random.randint(0, 2**32)),
    )

    # per_episode[key] = list of one scalar per eval episode
    per_episode = defaultdict(list)
    trajs = []
    renders = []

    total_episodes = num_eval_episodes + num_video_episodes
    for ep_idx in trange(total_episodes, desc='Eval', leave=False):
        is_video_ep = ep_idx >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        episode_return = 0.0
        render_frames = []
        action_queue = []

        while not done:
            # Time the policy call precisely (jax.block_until_ready ensures
            # the async XLA dispatch has completed before we stop the clock).
            t0 = time.perf_counter()
            raw_action = actor_fn(observations=observation)
            jax.block_until_ready(raw_action)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            # Action-chunk support: if the actor returns a chunk, queue all
            # actions and execute them one at a time. For 1-step actors this
            # queue is always empty at entry and gets one action pushed/popped.
            if len(action_queue) == 0:
                chunk = np.array(raw_action).reshape(-1, action_dim)
                action_queue.extend(chunk)

            action = action_queue.pop(0)
            next_observation, reward, terminated, truncated, info = env.step(
                np.clip(action, -1.0, 1.0)
            )
            done = terminated or truncated
            episode_return += float(reward)
            step += 1
            if not is_video_ep:
                per_episode['_step_inference_ms'].append(inference_ms)

            if is_video_ep and (step % video_frame_skip == 0 or done):
                render_frames.append(env.render().copy())

            observation = next_observation

        if is_video_ep:
            renders.append(np.array(render_frames))
            continue

        # --- Collect per-episode scalars ---
        per_episode['episode_return'].append(episode_return)
        per_episode['episode_length'].append(step)

        # Reduce the per-step inference times to a per-episode mean so the
        # final aggregate gives "mean latency per inference call" across all
        # steps and episodes.  Store and then remove the raw list.
        step_times = per_episode.pop('_step_inference_ms', [])
        if step_times:
            per_episode['inference_latency_ms'].append(float(np.mean(step_times)))

        # Flatten the terminal step's info dict (contains env-specific metrics
        # like 'success', 'distance_to_goal', etc.)
        for k, v in flatten(info).items():
            try:
                per_episode[k].append(float(v))
            except (TypeError, ValueError):
                pass  # skip non-scalar info fields

        trajs.append({'episode_return': episode_return, 'episode_length': step})

    stats = _aggregate(per_episode)
    return stats, trajs, renders
