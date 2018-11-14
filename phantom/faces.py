"""
Wraps dlib's face detectors and face encoder.

Support for other detectors could be added in the future.

Important links:

http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html
http://dlib.net/face_detector.py.html
http://dlib.net/face_landmark_detection.py.html
http://dlib.net/face_recognition.py.html
https://github.com/davisking/dlib
https://github.com/davisking/dlib-models
"""
# pylint: disable=E1101
#   - because of pylint not recognizing dlib.face_recognition_model_v1


import cv2
import dlib
import numpy as np


from pkg_resources import resource_filename


# paths for the model files
_path_shape_5p  = resource_filename("phantom", "models/shape_predictor_5_face_landmarks.dat")
_path_shape_68p = resource_filename("phantom", "models/shape_predictor_68_face_landmarks.dat")
_path_encoder   = resource_filename("phantom", "models/dlib_face_recognition_resnet_model_v1.dat")
# and we instance the models
face_detector       = dlib.get_frontal_face_detector()
face_encoder        = dlib.face_recognition_model_v1(_path_encoder)
shape_predictor_5p  = dlib.shape_predictor(_path_shape_5p)
shape_predictor_68p = dlib.shape_predictor(_path_shape_68p)


class Shape:
    """
    Represents the shape of a face, as returned from a facial landmark detector.

    :param points: ordered list of points, according to a landmark definition.
    """
    model = None  # subclasses override this

    def __init__(self, points):
        self.points = points
        self.dict = {}
        self._make_dict()
    
    def _make_dict(self):
        """
        Each subclass has to define this method to populate `self.dict`.
        """
        pass

    def _draw_lines(self, img, color, thick):
        """
        Subclasses must define the logic for drawing this shape over an image,
        using lines.
        """
        pass
    
    def _draw_points(self, img, color, thick):
        """
        Subclasses must define the logic for drawing this shape over an image,
        using points.
        """
        pass
    
    def _draw_numbers(self, img, color, thick):
        """
        Subclasses must define the logic for drawing this shape over an image,
        using numbers.
        """
        pass
    

class Shape68p(Shape):
    """
    68-point facial landmarks Shape object.
    """
    model = shape_predictor_68p

    def _make_dict(self):
        p = self.points
        self.dict = {
            "jawline":       p[0:17],
            "eyebrow_right": p[17:22],
            "eyebrow_left":  p[22:27],
            "nose_bridge":   p[27:31],
            "nose_tip":      p[31:36],
            "eye_right":     p[36:42],
            "eye_left":      p[42:48],
            "lips_top":      p[48:55] + p[64:59:-1],
            "lips_bottom":   p[54:60] + [p[48], p[60]] + p[67:63:-1],
            # "mouth":         p[],  # gotta check if it's usefull to implement this
        }


class FaceVault:
    pass



def _rect_to_tuple(r):
    """
    Helper function.

    Transforms a `dlib.rectangle` object into a tuple of (left, top, right,
    bottom) ints(longs).

    :param r: `dlib.rectangle` object
    :return: tuple of ints
    """
    return (r.left(), r.top(), r.right(), r.bottom())


def _tuple_to_rect(t):
    """
    Helper function.

    Transforms a tuple of (left, top, right, bottom) ints(longs) into a
    `dlib.rectangle` object.

    :param t: tuple of ints
    :return: `dlib.rectangle` object
    """
    return dlib.rectangle(*t)


def detect(img, upsample=1):
    """
    Detects faces present in an image.

    Wrapper of dlibs frontal face detector.

    :param img: numpy/cv2 image array
    :param upsample: int, number of times to upsample the image. Helps finding
        smaller faces
    :return: list of tuples (left, top, right, bottom) with each face location
    """
    return [_rect_to_tuple(r) for r in face_detector(img, upsample)]


def landmark(img, *, locations=None, model=Shape68p, upsample=1):
    """
    Detects the facial landmarks of each face present in an image.

    Wrapper of dlibs shape predictors.

    :param img: numpy/cv2 image array
    :param locations: list of tuples (left, top, right, bottom) with face 
        locations
    :param model: `Shape` subclass that defines a landmarking model.
    :param upsample: number of upsamples to use when locating faces (only used
        if `locations` is None)
    :return: list of `phantom.faces.Shape` objects, each describing the position
        and landmarks of every face
    """
    if locations is None:
        locations = detect(img, upsample)
    class_ = model
    model = class_.model  # TODO: might want to improve the names here
    shapelist = [model(img, _tuple_to_rect(loc)) for loc in locations]
    return [class_([(p.x, p.y) for p in face.parts()]) for face in shapelist]


def encode(img, *, locations=None, model=shape_predictor_68p, jitter=1):
    """
    Detects and encodes all the faces in an image.

    Wrapper of dlibs resnet facial encoder.

    :param img: numpy/cv2 image array
    :param locations: list of tuples (left, top, right, bottom) with face 
        locations
    :param model: shape predictor
    """
    if locations is None:
        locations = detect(img)
    shapelist = [model(img, _tuple_to_rect(loc)) for loc in locations]
    return [np.array(face_encoder.compute_face_descriptor(img, shape, jitter)) for shape in shapelist]


def compare(face1, face2):
    """
    Compares two face encodings (from dlib/`phantom.faces.encodings`).

    A distance under 0.6 means they are most certainly the same person (the
    lower the more certain). A distance slightly over 0.6 may mean they are the
    same person or not (pick a low enough epsilon). Distances over 0.6 mean the
    faces correspond to different people.

    :param face1: dlibs 128-long face encoding
    :param face2: dlibs 128-long face encoding
    :return: float, distance between `face1` and `face2`
    """
    return np.linalg.norm(face1 - face2)
