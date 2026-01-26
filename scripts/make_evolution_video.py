"""
Create a single "evolution" video from 3-stage videos:
  1_untrained-episode-0.mp4
  2_half_trained-episode-0.mp4
  3_fully_trained-episode-0.mp4

Default output is a side-by-side (3 columns) MP4 with labels, ideal for README.

Examples:
  python scripts/make_evolution_video.py --env-id merge-v0
  python scripts/make_evolution_video.py --all

Inputs are expected under:
  logs/videos/<env_id>/

Outputs go to:
  assets/videos/<env_id>_evolution.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


STAGES: List[Tuple[str, str]] = [
    ("1_untrained", "UNTRAINED"),
    ("2_half_trained", "HALF-TRAINED"),
    ("3_fully_trained", "FULLY TRAINED"),
]


@dataclass(frozen=True)
class Paths:
    env_id: str
    input_dir: str
    out_mp4: str

    @property
    def inputs(self) -> List[str]:
        return [os.path.join(self.input_dir, f"{prefix}-episode-0.mp4") for prefix, _ in STAGES]


def run(cmd: Sequence[str]) -> None:
    subprocess.run(cmd, check=True)


def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def ensure_files_exist(paths: Paths) -> None:
    missing = [p for p in paths.inputs if not os.path.isfile(p)]
    if missing:
        missing_str = "\n".join(f"- {m}" for m in missing)
        raise SystemExit(f"Missing input video(s) for {paths.env_id}:\n{missing_str}")


def build_filter_complex(font_size: int = 28) -> str:
    # Scale all to same height; keep aspect ratio; then hstack 3 videos.
    # Add text labels on each panel.
    # Note: font selection differs by OS; default font is ok for macOS/Linux.
    filters: List[str] = []
    for i, (_, label) in enumerate(STAGES):
        filters.append(
            f"[{i}:v]scale=-2:360,setsar=1,drawtext=text='{label}':x=10:y=10:fontsize={font_size}:fontcolor=white:box=1:boxcolor=0x00000099[v{i}]"
        )
    filters.append("[v0][v1][v2]hstack=inputs=3[outv]")
    return ";".join(filters)


def make_side_by_side(paths: Paths, crf: int = 23) -> None:
    ensure_files_exist(paths)
    os.makedirs(os.path.dirname(paths.out_mp4), exist_ok=True)

    filter_complex = build_filter_complex()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        paths.inputs[0],
        "-i",
        paths.inputs[1],
        "-i",
        paths.inputs[2],
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        "veryfast",
        paths.out_mp4,
    ]
    run(cmd)


def detect_env_ids(video_root: str) -> List[str]:
    if not os.path.isdir(video_root):
        return []
    envs: List[str] = []
    for name in sorted(os.listdir(video_root)):
        full = os.path.join(video_root, name)
        if os.path.isdir(full):
            envs.append(name)
    return envs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", help="Env folder under logs/videos (e.g. merge-v0)")
    parser.add_argument("--all", action="store_true", help="Process all env folders under logs/videos/")
    parser.add_argument("--video-root", default="logs/videos", help="Root folder containing per-env video folders")
    parser.add_argument("--outdir", default="assets/videos", help="Output directory")
    parser.add_argument("--crf", type=int, default=23, help="x264 quality (lower is better, larger file)")
    args = parser.parse_args()

    if not ffmpeg_exists():
        raise SystemExit("ffmpeg not found. Please install ffmpeg (e.g. via Homebrew: brew install ffmpeg).")

    if not args.all and not args.env_id:
        raise SystemExit("Provide --env-id <env_id> or --all")

    video_root = args.video_root
    outdir = args.outdir

    env_ids = detect_env_ids(video_root) if args.all else [args.env_id]
    if not env_ids:
        raise SystemExit(f"No env folders found under: {video_root}")

    for env_id in env_ids:
        input_dir = os.path.join(video_root, env_id)
        out_mp4 = os.path.join(outdir, f"{env_id}_evolution.mp4")
        paths = Paths(env_id=env_id, input_dir=input_dir, out_mp4=out_mp4)
        print(f"==> Building evolution video for {env_id}")
        make_side_by_side(paths, crf=args.crf)
        print(f"Saved: {out_mp4}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

