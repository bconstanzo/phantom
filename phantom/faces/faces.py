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
import pickle


from pkg_resources import resource_filename
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestCentroid
from sklearn import metrics


class _LazyStore:
    def __init__(self, d=None, r=None):
        if d is None:
            d = {}
        if r is None:
            r = {}
        self.dict = d
        self.reg  = r
    
    def register(self, key, initer, *args, **kwargs):
        self.reg[key] = (initer, args, kwargs)
    
    def get(self, key):
        try:
            return self.dict[key]
        except KeyError:
            pass  # don't want to nest try blocks
        # so we're here, without our key
        try:
            func, args, kwargs = self.reg[key]
            store = func(*args, **kwargs)
        except KeyError:
            raise KeyError("unregistered lazy key.")
        self.dict[key] = store
        return store


def _unpickle(path):
    """
    A simple wrapper to keep the loading of pickled models sane, and in line
    with the code style.
    """
    with open(path, "rb") as filehandle:
        obj = pickle.load(filehandle)
    return obj


# paths for the model files
_path_encoder   = resource_filename("phantom", "models/dlib_face_recognition_resnet_model_v1.dat")
if dlib.__version__.startswith("19.8"):
    _path_gender    = resource_filename("phantom", "models/phantom_gender_model_v1_dlib_19.8.dat")
    _path_gender_1b   = resource_filename("phantom", "models/phantom_gender_model_v1_dlib_19.8.dat")
    # TODO: cleanup and maybe retrain under dlib 19.8 or switch to scikit-learn SVM
else:
    _path_gender      = resource_filename("phantom", "models/phantom_gender_model_v1.dat")
    _path_gender_1b   = resource_filename("phantom", "models/phantom_gender_model_v1c.dat")
_path_age_model = resource_filename("phantom", "models/phantom_age_model_v1.dat")
_path_shape_5p  = resource_filename("phantom", "models/shape_predictor_5_face_landmarks.dat")
_path_shape_68p = resource_filename("phantom", "models/shape_predictor_68_face_landmarks.dat")
# and we instance the models
# scrub that -- lazy load them, with a lazy store
lazy_vars = _LazyStore()
lazy_vars.register(
    "face_detector", dlib.get_frontal_face_detector
)
lazy_vars.register(
    "face_encoder", dlib.face_recognition_model_v1, _path_encoder
)
lazy_vars.register(
    "age_model", _unpickle, _path_age_model
)
#face_encoder        = dlib.face_recognition_model_v1(_path_encoder)
lazy_vars.register(
    "gender_model", _unpickle, _path_gender
)
lazy_vars.register(
    "gender_model_1b", _unpickle, _path_gender_1b
)
#gender_model        = _unpickle(_path_gender)
lazy_vars.register(
    "shape_predictor_5p", dlib.shape_predictor, _path_shape_5p
)
#shape_predictor_5p  = dlib.shape_predictor(_path_shape_5p)
# Commented out until we resolve directly uploading the model file. 
lazy_vars.register(
    "shape_predictor_68p", dlib.shape_predictor, _path_shape_68p
)
#shape_predictor_68p = dlib.shape_predictor(_path_shape_68p)


class Shape:
    """
    Represents the shape of a face, as returned from a facial landmark detector.

    :param points: ordered list of points, according to a landmark definition.
    """

    def __init__(self, points):
        self.points = points
        self.dict = {}
        self._make_dict()
        self.model = None  # to be overridden by subclasses
    
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
        Subclasses can define the logic for drawing this shape over an image,
        using points, however a base implementation is provided.
        """
        for point in self.points:
            cv2.circle(img, point, thick, color, thickness=thick)
        return None
    
    def _draw_numbers(self, img, color, thick):
        """
        Subclasses must define the logic for drawing this shape over an image,
        using numbers.
        """
        pass


class Shape5p(Shape):
    """
    5-point facial landmarks Shape object.
    """
    def __init__(self, points):
        super().__init__(points)
        self.model = lazy_vars.get("shape_predictor_5p")

    def _make_dict(self):
        p = self.points
        self.dict = {
            "eye_left":   p[0:2],
            "eye_right":  p[2:4],
            "nose":      [p[4]],
        }
    
    def _draw_lines(self, img, color, thick):
        d = self.dict
        points = d["eye_left"] + d["nose"] + d["eye_right"][::-1]
        pairs = list(zip(points[:-1], points[1:]))
        # print(pairs)
        for point1, point2 in pairs:
            cv2.line(img, point1, point2, color, thickness=thick)
        return None


class Shape68p(Shape):
    """
    68-point facial landmarks Shape object.
    """
    def __init__(self, points):
        super().__init__(points)
        self.model = lazy_vars.get("shape_predictor_68p")

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
    
    def _draw_lines(self, img, color, thick):
        shape = self.dict
        color_ = color
        for key in shape:
            pairs = zip(shape[key][:-1], shape[key][1:])
            if key == "right_eye":
                color_ = (0, 0, 255)
            else:
                color_ = color
            for point1, point2 in pairs:
                cv2.line(img, point1, point2, color_, thickness=thick)
        return None


class Face:
    """
    This is a convenience class that helps to gather, in a single place, the
    encoding, landmarks, image, and origin of a face.

    In time we may use this class to transparently replace some conventions of
    phantom.

    :param landmark: a Shape object, that describes the landmarks of the face
    :param encoding: a 128-d vector encoding for a face
    :param image: an np.ndarray/cv2 image
    :param origin: path to a file
    :param landmarks: Shape object for the face
    :param location: location of a face within origin
    """
    # TODO: lazy loading of images from origin
    # TODO: extend the landmark parameter so it's a list/dict of Shapes?
    def __init__(self, encoding=None, image=None, origin=None, landmark=None, location=None):
        self.encoding = encoding
        self.image    = image
        self.origin   = origin
        self.landmark = landmark
        self.location = location
        self.tags     = {}


class Atlas:
    """
    A large grouping of facial encodings.

    :param encodings: list of Face objects, which at least have an encoding
    :param path: the path to which the atlas will persist on disk
    """
    def __init__(self, elements, path):
        self.elements  = elements
        self.clusters  = {}
        self.groups    = None
        self.grouped   = False
        self.path      = path
        self._db       = DBSCAN(eps=0.475, min_samples=2)
        self._ncc      = NearestCentroid()
        self._ncc_map  = {}
    
    def group(self):
        """
        Used clustering algorithms to group all the faces of the Atlas into
        distinct groups. These can later be used to match new faces, or compare
        to other Atlases.
        """
        if not self.elements:
            raise ValueError("Atlas doesn't have any elements to group")
        db = self._db
        faces = [f.encoding for f in self.elements]
        db = db.fit(faces)
        s_scores = metrics.silhouette_samples([e.encoding for e in self.elements], db.labels_)
        labeled_faces = (
            (f, l)
            for f, l, s in zip(faces, db.labels_, s_scores)
            if l >= 0 and s >= 0.20)
        # we'll only NCC the clusters that have well-defined silhuoette metric
        faces, labels = [list(x) for x in zip(*labeled_faces)]
        self._ncc.fit(faces, labels)
        self._ncc_map = { c: i for (i, c) in enumerate(self._ncc.classes_)}

    def predict(self, needles):
        """
        Quickly search the faces (encodings) in `needles` against the grouped
        centroids. This gives an extremely fast way of approximate search over
        the grouped faces.
        """
        ret = []
        ncc = self._ncc
        ncc_map = self._ncc_map
        results = ncc.predict(needles)
        for class_, needle in zip(results, needles):
            idx = ncc_map[class_]
            centroid = self._ncc.centroids_[idx]
            distance = compare(needle, centroid)
            ret.append((class_, distance))
        return ret

    
    def load(self):
        """
        Loads an Atlas from disk, read from `self.path`.
        """
        with open(self.path, "rb") as fhandle:
            new_dict = pickle.load(fhandle)
        self.__dict__.update(new_dict)
    
    def save(self):
        """
        Persists the Atlas on disk, written to `self.path`.
        """
        with open(self.path, "wb") as fhandle:
            pickle.dump(self.__dict__, fhandle)



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


def detect(img, *, upsample=1):
    """
    Detects faces present in an image.

    Wrapper of dlibs frontal face detector.

    :param img: numpy/cv2 image array
    :param upsample: int, number of times to upsample the image. Helps finding
        smaller faces
    :return: list of tuples (left, top, right, bottom) with each face location
    """
    face_detector = lazy_vars.get("face_detector")
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
        locations = detect(img, upsample=upsample)
    class_ = model
    shaper = model([(i, i) for i in range(68)])
    model = shaper.model  # TODO: might want to improve the names here
    shapelist = [model(img, _tuple_to_rect(loc)) for loc in locations]
    return [class_([(p.x, p.y) for p in face.parts()]) for face in shapelist]


def normalize_landmark(shape):
    """
    Takes a Shape object and normalizes its values to the [-1, 1] range.

    :param shape: Shape object (must have .points attribute)
    :return: np.ndarray with the points normalized (rows of [x, y] values)
    """
    vector  = np.array(shape.points, dtype=np.float32)
    min_v   = np.min(vector, axis=0)  # min for both x and y at the same time
    vector -= min_v
    vector /= np.max(vector, axis=0)
    vector -= np.array([0.5, 0.5]) 
    vector *= 2  # keeps everything in range [-1, 1]
    return vector


def encode(img, *, locations=None, model=Shape68p, jitter=1):
    """
    Detects and encodes all the faces in an image.

    Wrapper of dlibs resnet facial encoder.

    :param img: numpy/cv2 image array
    :param locations: list of tuples (left, top, right, bottom) with face 
        locations
    :param model: shape predictor
    :param jitter: an integer number of times to scramble the image a bit, and
        re-run the encoding. Higher jitter makes for slightly better encodings,
        though it slows down the encoding.
    """
    if locations is None:
        locations = detect(img)
    shaper = model([(i, i) for i in range(68)])
    model = shaper.model  # once again this shadowing...
    face_encoder = lazy_vars.get("face_encoder")
    shapelist = [model(img, _tuple_to_rect(loc)) for loc in locations]
    return [np.array(face_encoder.compute_face_descriptor(img, shape, jitter)) for shape in shapelist]


def compare(face1, face2):
    """
    Compares two face encodings (from dlib/`phantom.faces.encodings`).

    A distance under 0.6 means the faces correspond to the same person. A
    distance slightly over 0.6 (+epsilon) means it could be the same person, for
    a low enough epsilon. Distances over 0.6 mean the faces are of different
    people.

    :param face1: dlibs 128-long face encoding
    :param face2: dlibs 128-long face encoding
    :return: float, distance between `face1` and `face2`
    """
    return np.linalg.norm(face1 - face2)


def estimate_age(face):
    """
    Estimates the age of a person based on a facial encoding.

    :param face: dlibs 128-long face encoding
    """
    age_model = lazy_vars.get("age_model")
    face = face.reshape(1, -1)
    return age_model.predict(face)


def estimate_gender(face, *, multi=False):
    """
    Estimates a characteristic based on the face that is passed.

    :param face: dlibs 128-long face encoding
    :return: float, estimated gender. The gender model has been trained as
        value 1 for females, and -1 for males. So, a value of -0.5 means "mainly
        male" and can be considered as such. Values between -0.3 and 0.3 mean
        the model is not certain enough, and should be considered as "unknown"
        or "uncertain"
    """
    vector = dlib.vector(face)
    gender_model = lazy_vars.get("gender_model")
    if multi:
        gender_model2 = lazy_vars.get("gender_model_1b")
        return (gender_model(vector), gender_model2(vector))
    return gender_model(vector)
