"""Find the buffer size that lets the loop keep up with the camera.

Needs the camera. Needs no display, no model, no face and nobody sitting still:
it reads frames, sleeps to imitate the loop's work, and counts.

    uv run --no-sync python scripts/probe_capture_rate.py
    uv run --no-sync python scripts/probe_capture_rate.py --work-ms 25 --frames 90

The question. `CAP_PROP_BUFFERSIZE` of 1 came from the calibration sessions,
where a reader that went idle and came back got a four frame old still. That is
a real finding and it is about **idle-then-read**. A loop that reads
continuously has the opposite problem: with a single buffer the driver has
nowhere to put the next frame while userspace is busy, so a frame that arrives
during the loop's own work is dropped and the read waits for the one after it.
The loop then runs at half the camera's rate however fast it is.

Measured on g7: 67 ms per frame against 33.3 ms of camera interval, with only
25 ms of work. Exactly two intervals, and 14.8 fps against a camera doing 30.

So this sweeps the buffer size against a simulated work time and reports what
actually arrives. It answers whether one line fixes it or whether the loop
needs the capture thread named as the upgrade in the phase 5 plan.
"""

import argparse
import time

import cv2 as cv
from loguru import logger as lg
import numpy as np

from abyss.video.capture import MJPG_FOURCC

DEFAULT_DEVICE = 0
CAPTURE_SIZE = (1280, 720)
DEFAULT_BUFFERS = "1,2,3,4"
DEFAULT_WORK_MS = 25.0
DEFAULT_FRAMES = 90
WARMUP_FRAMES = 10

CAMERA_FPS = 30.0
"""What the camera advertises, and the ceiling any of this can reach."""


def probe(device: int, buffers: int, work_ms: float, frames: int) -> dict:
    """Read frames at one buffer size, imitating the loop's work between them.

    Args:
        device: Camera device index.
        buffers: What to set ``CAP_PROP_BUFFERSIZE`` to.
        work_ms: Milliseconds to burn between reads, standing in for the
            landmarker, the render and the sink.
        frames: Frames to time, past the warm-up.

    Returns:
        A row of results for this buffer size.
    """
    capture = cv.VideoCapture(device)
    capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*MJPG_FOURCC))
    capture.set(cv.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
    capture.set(cv.CAP_PROP_BUFFERSIZE, buffers)
    got = int(capture.get(cv.CAP_PROP_BUFFERSIZE))

    reads: list[float] = []
    started = 0.0
    for idx in range(frames + WARMUP_FRAMES):
        mark = time.perf_counter()
        ok, _frame = capture.read()
        read_ms = (time.perf_counter() - mark) * 1000
        if not ok:
            break
        # Burn the work time the real loop would spend, without doing it.
        busy_until = time.perf_counter() + work_ms / 1000
        while time.perf_counter() < busy_until:
            pass
        if idx == WARMUP_FRAMES - 1:
            started = time.perf_counter()
        if idx >= WARMUP_FRAMES:
            reads.append(read_ms)
    elapsed = time.perf_counter() - started
    capture.release()

    achieved = len(reads) / elapsed if elapsed > 0 else 0.0
    return {
        "asked": buffers,
        "got": got,
        "read_ms": float(np.median(reads)) if reads else 0.0,
        "fps": achieved,
        "of_camera": achieved / CAMERA_FPS,
    }


def main() -> None:
    """Sweep the buffer size and report what arrives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE)
    parser.add_argument("--buffers", default=DEFAULT_BUFFERS)
    parser.add_argument("--work-ms", type=float, default=DEFAULT_WORK_MS)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    args = parser.parse_args()

    lg.info(
        f"Imitating {args.work_ms:.0f} ms of work per frame, "
        f"against a camera advertising {CAMERA_FPS:.0f} fps"
    )
    lg.info(f"{'asked':>5} {'got':>4} {'read ms':>8} {'fps':>7} {'of camera':>10}")
    for buffers in [int(b) for b in args.buffers.split(",")]:
        row = probe(args.device, buffers, args.work_ms, args.frames)
        lg.info(
            f"{row['asked']:>5} {row['got']:>4} {row['read_ms']:>8.1f} "
            f"{row['fps']:>7.1f} {row['of_camera']:>9.0%}"
        )


if __name__ == "__main__":
    main()
