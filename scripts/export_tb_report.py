from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


COMMON_TAGS = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/loss",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/entropy_loss",
    "time/fps",
]


@dataclass(frozen=True)
class ScalarSeries:
    steps: List[int]
    values: List[float]


def find_event_dirs(logdir: str) -> List[str]:
    event_dirs: List[str] = []
    for root, dirs, files in os.walk(logdir):
        if any(f.startswith("events.out.tfevents") for f in files):
            event_dirs.append(root)
    return sorted(set(event_dirs))


def guess_env_id_from_path(path: str, logdir: str) -> str:
    rel = os.path.relpath(path, logdir)
    first = rel.split(os.sep)[0]
    return first if first and first != "." else "unknown-env"


def load_scalars(event_dir: str, tags: List[str]) -> Dict[str, ScalarSeries]:
    # size_guidance keeps memory reasonable even for long runs
    ea = EventAccumulator(event_dir, size_guidance={"scalars": 50_000})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out: Dict[str, ScalarSeries] = {}
    for tag in tags:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        steps = [int(e.step) for e in events]
        values = [float(e.value) for e in events]
        if steps:
            out[tag] = ScalarSeries(steps=steps, values=values)
    return out


def plot_series(series: ScalarSeries, title: str, xlabel: str = "timesteps") -> plt.Figure:
    fig = plt.figure(figsize=(9, 4.5), dpi=160)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(series.steps, series.values, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def save_fig(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def export_env_plots(env_id: str, runs: List[str], outdir: str) -> List[str]:
    # For each env, pick the "best" run directory as the deepest/most recent one
    # (common structure: <env_id>/<algo>_<seed>/)
    # If multiple exist, export from all and suffix by run name.
    exported: List[str] = []
    for event_dir in runs:
        run_name = os.path.basename(event_dir)
        scalars = load_scalars(event_dir, COMMON_TAGS)
        for tag, series in scalars.items():
            safe_tag = tag.replace("/", "__")
            out_path = os.path.join(outdir, "tb", env_id, f"{safe_tag}__{run_name}.png")
            fig = plot_series(series, title=f"{env_id} • {tag} • {run_name}")
            save_fig(fig, out_path)
            exported.append(out_path)
    return exported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs/tensorboard", help="TensorBoard log root directory")
    parser.add_argument("--outdir", default="assets", help="Output directory for PNG plots")
    args = parser.parse_args()

    logdir = os.path.abspath(args.logdir)
    outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(logdir):
        raise SystemExit(f"Log directory not found: {logdir}")

    event_dirs = find_event_dirs(logdir)
    if not event_dirs:
        raise SystemExit(f"No TensorBoard event files found under: {logdir}")

    by_env: Dict[str, List[str]] = {}
    for d in event_dirs:
        env_id = guess_env_id_from_path(d, logdir)
        by_env.setdefault(env_id, []).append(d)

    exported_all: List[str] = []
    for env_id, runs in sorted(by_env.items()):
        exported_all.extend(export_env_plots(env_id=env_id, runs=runs, outdir=outdir))

    print(f"Exported {len(exported_all)} plot(s) into: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

