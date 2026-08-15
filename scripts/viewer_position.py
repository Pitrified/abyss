"""Track a viewer's eye position through a recorded clip.

Writes a CSV and a plot per clip under ``AbyssPaths.cache_fol``, both checkable
on a headless box. Run it directly:

    uv run --no-sync python scripts/viewer_position.py
    uv run --no-sync python scripts/viewer_position.py face03_zoom.mp4 --ipd-mm 63

With no argument it processes every ``face*.mp4`` in ``AbyssPaths.pose_fol``.
"""

import argparse
import csv
from pathlib import Path

import cv2 as cv
from loguru import logger as lg
import matplotlib as mpl
from matplotlib import pyplot as plt
from mediapipe.tasks.python.vision.core.image import Image
from mediapipe.tasks.python.vision.core.image import ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode as VisionRunningMode,
)
import numpy as np
from pose_tools.landmark.face import FaceLandmarkerFrame
from pose_tools.landmark.model_manager import ModelManager
from pose_tools.video.frame import Frame

from abyss.config.camera import FrameGeometry
from abyss.config.viewer import DEFAULT_IPD_M
from abyss.config.viewer import ViewerConfig
from abyss.params.abyss_devices import get_camera
from abyss.params.abyss_params import get_abyss_paths
from abyss.viewer.eye_position import EyeSample
from abyss.viewer.eye_position import estimate_head_scale
from abyss.viewer.eye_position import extract_eye_sample
from abyss.viewer.eye_position import eye_position_m
from abyss.viewer.smoothing import PositionSmoother

mpl.use("Agg")  # headless box: no display, so figures go to files

CLIP_CAMERA = "unknown_clip"
"""The sample clips have no known camera, so the fallback intrinsics apply."""

CSV_FIELDS = [
    "idx",
    "msec",
    "has_face",
    "iris_u_px",
    "iris_v_px",
    "ipd_px",
    "yaw_deg",
    "x_m",
    "y_m",
    "z_m",
    "x_smooth_m",
    "y_smooth_m",
    "z_smooth_m",
]


def collect_samples(clip: Path) -> tuple[list[EyeSample | None], FrameGeometry]:
    """Run the landmarker over a clip and collect the raw samples.

    Args:
        clip: Path to the video file.

    Returns:
        The per-frame samples, ``None`` where no face was found, and the frame
        geometry the clip turned out to have.
    """
    capture = cv.VideoCapture(str(clip))
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv.CAP_PROP_FPS)
    # The frame size comes from the frames, never from the config: the focal
    # length follows the frame height, so a stale resolution rescales depth.
    geometry = FrameGeometry(camera=get_camera(CLIP_CAMERA), width=width, height=height)
    lg.info(f"{clip.name}: {width}x{height} @ {fps:.2f} fps")

    model_path = ModelManager().ensure_model("face_landmarker")
    samples: list[EyeSample | None] = []
    landmarker_kwargs = {
        "running_mode": VisionRunningMode.VIDEO,
        "num_faces": 1,
        "output_facial_transformation_matrixes": True,
    }
    with FaceLandmarkerFrame(model_path, landmarker_kwargs) as landmarker:
        idx = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
            # Timestamps must be strictly increasing for VIDEO mode.
            msec = idx * 1000.0 / fps
            frame = Frame(
                image=Image(image_format=ImageFormat.SRGB, data=rgb),
                msec=msec,
                idx=idx,
            )
            result = landmarker.detect(frame)
            samples.append(extract_eye_sample(result, geometry, idx, msec))
            idx += 1
    capture.release()

    found = sum(s is not None for s in samples)
    lg.info(f"{clip.name}: {found}/{len(samples)} frames with a face")
    return samples, geometry


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write the per-frame track.

    Args:
        path: Destination CSV path.
        rows: One dict per frame, keyed by :data:`CSV_FIELDS`.
    """
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lg.info(f"Wrote {path}")


def write_plot(path: Path, rows: list[dict], title: str) -> None:
    """Plot the three axes over time, raw against smoothed.

    Args:
        path: Destination image path.
        rows: One dict per frame, keyed by :data:`CSV_FIELDS`.
        title: Figure title.
    """
    tracked = [r for r in rows if r["has_face"]]
    seconds = np.array([r["msec"] for r in tracked]) / 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, axis in zip(axes, "xyz", strict=True):
        raw = np.array([r[f"{axis}_m"] for r in tracked])
        smooth = np.array([r[f"{axis}_smooth_m"] for r in tracked])
        ax.plot(seconds, raw, linewidth=0.8, alpha=0.5, label="raw")
        ax.plot(seconds, smooth, linewidth=1.5, label="smoothed")
        ax.set_ylabel(f"{axis} (m)")
        ax.grid(visible=True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    lg.info(f"Wrote {path}")


def process(clip: Path, out_fol: Path, ipd_m: float) -> None:
    """Track one clip and write its artefacts.

    Args:
        clip: Path to the video file.
        out_fol: Folder to write the CSV and the plot into.
        ipd_m: Viewer interpupillary distance in metres.
    """
    samples, geometry = collect_samples(clip)
    viewer = ViewerConfig(name="cli", ipd_m=ipd_m)
    found = [s for s in samples if s is not None]
    if not found:
        lg.warning(f"{clip.name}: no face anywhere, nothing to write")
        return

    head_scale = estimate_head_scale(found, geometry, viewer)
    smoother = PositionSmoother()

    rows: list[dict] = []
    for sample in samples:
        if sample is None:
            held = smoother.hold()
            rows.append(
                {
                    "idx": len(rows),
                    "msec": 0.0,
                    "has_face": False,
                    "iris_u_px": "",
                    "iris_v_px": "",
                    "ipd_px": "",
                    "yaw_deg": "",
                    "x_m": "",
                    "y_m": "",
                    "z_m": "",
                    "x_smooth_m": "" if held is None else held[0],
                    "y_smooth_m": "" if held is None else held[1],
                    "z_smooth_m": "" if held is None else held[2],
                }
            )
            continue
        position = eye_position_m(sample, geometry, head_scale)
        smoothed = smoother.update(position)
        rows.append(
            {
                "idx": sample.idx,
                "msec": round(sample.msec, 2),
                "has_face": True,
                "iris_u_px": round(sample.u_px, 2),
                "iris_v_px": round(sample.v_px, 2),
                "ipd_px": round(sample.ipd_px, 2),
                "yaw_deg": round(sample.yaw_deg, 2),
                "x_m": round(float(position[0]), 5),
                "y_m": round(float(position[1]), 5),
                "z_m": round(float(position[2]), 5),
                "x_smooth_m": round(float(smoothed[0]), 5),
                "y_smooth_m": round(float(smoothed[1]), 5),
                "z_smooth_m": round(float(smoothed[2]), 5),
            }
        )

    depths = np.array([r["z_m"] for r in rows if r["has_face"]])
    lg.info(
        f"{clip.name}: depth {depths.min():.3f}-{depths.max():.3f} m "
        f"(mean {depths.mean():.3f})"
    )

    stem = clip.stem
    write_csv(out_fol / f"{stem}_eye.csv", rows)
    write_plot(out_fol / f"{stem}_eye.png", rows, f"{stem}: eye position")


def main() -> None:
    """Parse arguments and process the requested clips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "clip",
        nargs="?",
        help="clip name inside the pose folder, or a path. Default: every face*.mp4",
    )
    parser.add_argument(
        "--ipd-mm",
        type=float,
        default=DEFAULT_IPD_M * 1000,
        help="viewer interpupillary distance in millimetres",
    )
    args = parser.parse_args()

    paths = get_abyss_paths()
    out_fol = paths.cache_fol / "viewer"
    out_fol.mkdir(parents=True, exist_ok=True)

    if args.clip:
        candidate = Path(args.clip)
        clips = [candidate if candidate.exists() else paths.pose_fol / args.clip]
    else:
        clips = sorted(paths.pose_fol.glob("face*.mp4"))

    if not clips:
        lg.error(f"No clips found in {paths.pose_fol}")
        return

    for clip in clips:
        if not clip.exists():
            lg.error(f"Missing clip: {clip}")
            continue
        process(clip, out_fol, args.ipd_mm / 1000)


if __name__ == "__main__":
    main()
