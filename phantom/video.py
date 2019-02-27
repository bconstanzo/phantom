"""
Miscelanous utilities that allow to easily work with video.

Mostly wraps OpenCVs classes and functions to give an easy interface. Mostly
aimed at making the examples easier to understand.s
"""

# still a work in progress
# consider reading:
# * https://stackoverflow.com/questions/3431434/video-stabilization-with-opencv/47483926#47483926
# * https://www.learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/?ck_subscriber_id=272167184
# * Optical flow
import cv2


from itertools import cycle


def play_list(images, delay=10, esc_key="q", title="phantom playback"):
    """
    Plays a sequence of images as a video, using cv2.imshow().

    WARNING: as it stands today, it'll crash when used from the console. This is
    not the intended use case. We'll probably port this to matplotlib in the
    future.

    :param images: list of np.ndarray/cv2 images.
    :param delay: how many miliseconds to way for a keypress.
    :param esc_key: (one char string) which keypress will end the playback.
    :param title: title for the playback window
    """
    images = cycle(images)
    while True:
        frame = next(images)
        cv2.imshow(title, frame)
        key = cv2.waitKey(delay)
        if key == ord(esc_key):
            break