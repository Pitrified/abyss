"""Test the video acquisition from the camera."""

import cv2 as cv

from abyss.video.load import VideoFrameIterator


def test_from_camera() -> None:
    """Test the video acquisition from the camera."""
    try:
        # with VideoFrameIterator() as vid_iter:
        with VideoFrameIterator(max_frame_count=20) as vid_iter:
            for frame in vid_iter:
                # show the frame
                cv.imshow("frame", frame.to_opencv())
                # set the title of the window
                cv.setWindowTitle("frame", f"{frame}")
                # wait for the user to press 'q' to quit
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv.destroyAllWindows()


if __name__ == "__main__":
    test_from_camera()
