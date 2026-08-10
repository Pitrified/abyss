"""Show the camera feed, to check that video acquisition works.

Manual script: needs a camera and a display, so it cannot run headless.
"""

import cv2 as cv
from pose_tools.video.load import VideoFrameIterator


def show_camera_feed(max_frame_count: int = 20) -> None:
    """Show frames from the camera until they run out or 'q' is pressed.

    Args:
        max_frame_count: How many frames to read before stopping.
    """
    try:
        with VideoFrameIterator(max_frame_count=max_frame_count) as vid_iter:
            for frame in vid_iter:
                cv.imshow("frame", frame.to_opencv())
                cv.setWindowTitle("frame", f"{frame}")
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
    except cv.error as e:
        print(f"OpenCV failed, is a camera and a display available? {e}")
    finally:
        cv.destroyAllWindows()


if __name__ == "__main__":
    show_camera_feed()
