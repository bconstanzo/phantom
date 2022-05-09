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


from concurrent.futures import ThreadPoolExecutor
from itertools import cycle


def threaded_videocap_read(cap):
    return cap.read()


class VideoThreaded:
    def __init__(self, path):
        """
        Iterable video class, it allows writing code of the form:

            >>> vid = phantom.video.Video(path_to_file)
            >>> for frame in vid:
            >>>     process(frame)
        
        Offloads reading to a separate thread. When reading from files it may
        speed processing quite a bit.

        :param path: path to the video.
        """
        self.path = path
    
    def __iterator(self):
        with ThreadPoolExecutor(1) as pool:
            cap = cv2.VideoCapture(self.path)
            status, frame = cap.read()
            while status:
                future_result = pool.submit(threaded_videocap_read, cap)
                yield frame
                # status, frame = cap.read()
                status, frame = future_result.result()
            cap.release()
    
    def __iter__(self):
        return self.__iterator()



class Video:
    def __init__(self, path):
        """
        Iterable video class, it allows writing code of the form:

            >>> vid = phantom.video.Video(path_to_file)
            >>> for frame in vid:
            >>>     process(frame)
        
        :param path: path to the video.
        """
        self.path = path
    
    def __iterator(self):
        cap = cv2.VideoCapture(self.path)
        status, frame = cap.read()
        while status:
            yield frame
            status, frame = cap.read()
        cap.release()
    
    def __iter__(self):
        return self.__iterator()


def play_list(images, delay=10, esc_key="q", title="phantom playback"):
    """
    Plays a sequence of images as a video, using cv2.imshow().

    * NOT IMPLEMENTED*

    :param images: list of np.ndarray/cv2 images.
    :param delay: how many miliseconds to way for a keypress.
    :param esc_key: (one char string) which keypress will end the playback.
    :param title: title for the playback window
    """
    pass
