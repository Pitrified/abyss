"""Load a video from a file or a camera.

https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
"""

from pathlib import Path
from typing import Generator, Self

import cv2 as cv
from loguru import logger as lg

from abyss.video.frame import Frame


class VideoFrameIterator:
    def __init__(
        self,
        in_vid_path: Path | None = None,
        keep_every_nth_frame: int = 1,
        max_frame_count: int = 0,
    ) -> None:
        """Initialize the video frame iterator.

        Args:
            in_vid_path: Input video file. Set to None to use the camera.
            keep_every_nth_frame: Keep every nth frame in the video.
            max_frame_count: Maximum number of frames to extract.
                Set to 0 to extract all frames or until the end of the stream.
        """
        self.in_vid_path = in_vid_path
        self.keep_every_nth_frame = keep_every_nth_frame
        self.max_frame_count = max_frame_count
        self.cap: cv.VideoCapture | None = None
        self.feed_count = 0
        self.yield_count = 0

    def __iter__(self) -> Generator[Frame, None, None]:
        """Return the iterator object."""
        if self.cap is None:
            raise ValueError("Video file or camera feed not opened.")

        success = True
        while success and (
            self.max_frame_count == 0 or self.yield_count < self.max_frame_count
        ):
            success, frame = self.cap.read()
            if not success:
                break

            pos_msec = self.cap.get(cv.CAP_PROP_POS_MSEC)

            if self.feed_count % self.keep_every_nth_frame == 0:
                lg.debug(f"Yielding frame {self.yield_count} at {pos_msec:.2f} ms")
                yield Frame.from_opencv(frame, pos_msec, self.yield_count)
                self.yield_count += 1

            self.feed_count += 1

    def __enter__(self) -> Self:
        """Open the video file or camera feed."""
        if self.in_vid_path is None:
            self.cap = cv.VideoCapture(0)
        else:
            self.cap = cv.VideoCapture(str(self.in_vid_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the video file or camera feed."""
        if self.cap is not None:
            lg.debug("Releasing video capture")
            self.cap.release()


def list_video_frames(
    in_vid_path: Path,
    keep_every_nth_frame: int = 1,
    max_frame_count: int = 0,
) -> list[Frame]:
    """Extract frames from video, return them as a list.

    Args:
        in_vid_path: Input video file.
        keep_every_nth_frame: Keep every nth frame in the video.
        max_frame_count: Maximum number of frames to extract.
            Set to 0 to extract all frames.
    """
    frames: list[Frame] = []

    with VideoFrameIterator(
        in_vid_path,
        keep_every_nth_frame=keep_every_nth_frame,
        max_frame_count=max_frame_count,
    ) as frame_iterator:
        for frame in frame_iterator:
            frames.append(frame)

    return frames
