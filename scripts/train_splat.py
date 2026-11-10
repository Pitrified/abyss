"""Train a Gaussian splat on g7 with gsplat, and place it for the phone page.

    uv run --no-sync python scripts/train_splat.py ~/data/splat_train/360_v2/room_q room

Three steps, each skipped when its output is already there:

1. Train with gsplat's ``examples/simple_trainer.py`` at its defaults, in
   gsplat's own environment (``--gsplat``, default ``~/repos/gsplat``), into
   ``~/data/splat_train/runs/<name>_<steps>``.
2. Convert the trained PLY to SPZ with Spark, into ``AbyssPaths.splats_fol``.
3. Write the placement file beside it, with one capture camera at the viewer's
   eye (:func:`abyss.ui.splat.placement_at_camera`); ``--camera`` picks which,
   and ``--scale`` sets metres per unit.

The scene folder is COLMAP's layout: ``images/`` and ``sparse/0/``. gsplat's
loader reads the images at the size they are, scaling the intrinsics to match,
so a folder whose ``images`` holds downscaled copies trains at that size.

The camera poses are read back through gsplat's own loader, so the placement
uses the frame the splat was trained in. That frame's z is only roughly up, so
up is the cameras' mean up (:func:`abyss.ui.splat.capture_up`).

Not a test: it needs the GPU and takes tens of minutes.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

from loguru import logger as lg
import numpy as np

from abyss.params.abyss_params import get_abyss_paths
from abyss.ui.splat import capture_up
from abyss.ui.splat import pick_camera
from abyss.ui.splat import placement_at_camera
from abyss.ui.spz import ply_to_spz

TRAIN_ROOT = Path.home() / "data" / "splat_train" / "runs"
GSPLAT = Path.home() / "repos" / "gsplat"
CUDA_HOME = "/usr/local/cuda"

POSES_PY = """
import json, sys
from datasets.colmap import Parser
parser = Parser(sys.argv[1], factor=1, normalize=True, test_every=8)
print(json.dumps(parser.camtoworlds[:, :3, :].tolist()))
"""
"""Run in gsplat's environment: camera-to-world matrices in the training frame."""


def gsplat_env() -> dict[str, str]:
    """Return the environment gsplat's CUDA code runs in."""
    return {
        **os.environ,
        "CUDA_HOME": CUDA_HOME,
        "PATH": f"{CUDA_HOME}/bin:{os.environ['PATH']}",
    }


def train(gsplat: Path, scene: Path, result: Path, steps: int) -> Path:
    """Train with gsplat's default trainer, unless the PLY is already there.

    Args:
        gsplat: The gsplat clone, with its ``.venv``.
        scene: The COLMAP scene folder.
        result: The run folder.
        steps: Training steps.

    Returns:
        The trained PLY.
    """
    ply = result / "ply" / f"point_cloud_{steps - 1}.ply"
    if ply.exists():
        lg.info(f"Trained already: {ply}")
        return ply
    final = str(steps)
    subprocess.run(  # noqa: S603 - our own paths and numbers, no shell
        [
            gsplat / ".venv" / "bin" / "python",
            "simple_trainer.py",
            "default",
            "--data-dir", scene,
            "--data-factor", "1",
            "--result-dir", result,
            "--max-steps", final,
            "--eval-steps", final,
            "--save-steps", final,
            "--save-ply",
            "--ply-steps", final,
            "--disable-viewer",
            "--disable-video",
        ],
        cwd=gsplat / "examples",
        env=gsplat_env(),
        check=True,
    )  # fmt: skip
    return ply


def camera_poses(gsplat: Path, scene: Path) -> np.ndarray:
    """Read the cameras through gsplat's loader, in its training frame.

    Args:
        gsplat: The gsplat clone.
        scene: The COLMAP scene folder.

    Returns:
        ``(N, 3, 4)`` camera-to-world matrices, OpenCV axes, z-up world.
    """
    out = subprocess.run(  # noqa: S603 - our own paths, no shell
        [gsplat / ".venv" / "bin" / "python", "-c", POSES_PY, scene],
        cwd=gsplat / "examples",
        env=gsplat_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    return np.array(json.loads(out.stdout.strip().splitlines()[-1]))


def main() -> None:
    """Train, convert and place one scene."""
    parser = argparse.ArgumentParser(description="Train, convert and place a splat.")
    parser.add_argument("scene", type=Path, help="COLMAP folder: images/, sparse/0/")
    parser.add_argument("name", help="Splat name in the splats folder")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--gsplat", type=Path, default=GSPLAT)
    parser.add_argument("--camera", type=int, help="Capture camera at the eye")
    parser.add_argument("--scale", type=float, default=1.0, help="Metres per unit")
    args = parser.parse_args()

    result = TRAIN_ROOT / f"{args.name}_{args.steps}"
    ply = train(args.gsplat, args.scene, result, args.steps)

    splats = get_abyss_paths().splats_fol
    splats.mkdir(parents=True, exist_ok=True)
    spz = splats / f"{args.name}.spz"
    if not spz.exists() or spz.stat().st_mtime < ply.stat().st_mtime:
        ply_to_spz(ply, spz)

    poses = camera_poses(args.gsplat, args.scene)
    centres, forwards = poses[:, :, 3], poses[:, :, 2]
    up = capture_up(-poses[:, :, 1])
    picked = pick_camera(centres, forwards, up)
    camera = picked if args.camera is None else args.camera
    placement = placement_at_camera(
        centres[camera], forwards[camera], up, scale=args.scale
    )
    placement_file = splats / f"{args.name}.json"
    placement_file.write_text(json.dumps({"file": spz.name, **placement}) + "\n")
    lg.info(f"Camera {camera} at the eye; wrote {placement_file}")


if __name__ == "__main__":
    main()
