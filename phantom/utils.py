"""
Utility functions that help in working with the different faces, but are not
essential. Includes functions for drawing landmarks, showing OpenCV images with
matplotlib, etc.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def color_correct(base, dest, baselm):
    """
    Modifies `dest` colors to approximately match `base`s.

    :param base: reference image (cv2/np.ndarray)
    :param dest: image to modify (cv2/np.ndarray)
    :param baselm: facial landmarks from face_recognition.face_landmarks() of
        the reference image (`base`)
    :return: color-corrected `dest`-copy (cv2/np.ndarray)
    """
    leye = np.mean(baselm["left_eye"], axis=0)
    reye = np.mean(baselm["right_eye"], axis=0)
    amount = int(0.6 * np.linalg.norm(leye - reye))
    if (amount % 2 == 0):
        amount += 1
    blur1 = cv2.GaussianBlur(base, (amount, amount), 0).astype(np.float32)
    blur2 = cv2.GaussianBlur(dest, (amount, amount), 0).astype(np.float32)
    blur2 += 128 * (blur2 <= 1.0)
    ret = dest.copy().astype(np.float32)
    ret = (ret * (blur1 / blur2)).astype(np.uint8)
    ret = np.minimum(np.maximum(ret, 0), 255)
    return ret


# TODO: implement a Landmark class to delete this function
def _landmark_points(landmarks):
    """
    Quick comprehension to get the points of a facial landmark in list form.

    FIXME: :param landmarks: a dict of facial landmarks, an element of facial_recognition.face_landmarks.
    :return: list of points, in order.
    """
    return [e 
            for k in landmarks
            for e in landmarks[k]]


def _draw_face_line(img, shape, color=(0, 255, 0), thick=2):
    """
    Semi-private function that draws the landmarks of a face over an image,
    using lines.
    
    :param img: image on which to draw
    :param shape: `phantom.faces.Shape` object describing a face in img
    :param color: BGR color for drawing
    :param thick: line thickness
    """
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


def _draw_face_numbers(img, shape, color=(0, 255, 0), thick=2):
    pass


def _draw_face_points(img, shape, color=(0, 255, 0), thick=2):
    """
    Semi-private funciton that draws the landmarks of a face over an image,
    using points.
    
    :param img: image on which to draw
    :param shape: `phantom.faces.Shape` object describing a face in img
    :param color: BGR color for drawing
    :param thick: line thickness, also used as circle radius
    """
    for key in shape:
        for point in shape[key]:
            cv2.circle(img, point, thick, color, thickness=thick)
    return None


def draw_faces(img, faces, *, color=(0, 255, 0), thick=2, mode="line", on_copy=True):
    """
    Draw a set of faces over img.

    :param img: image on which to draw
    :param shape: `phantom.faces.Shape` object describing a face in img
    :param color: BGR color for drawing
    :param thick: line thicknes or circle radius, depending on mode
    :param mode: draw lines or points over the landmark points
    :param on_copy: whether to work on a copy or `img` directly
    """
    if on_copy:
        img_ = img.copy()
    else:
        img_ = img
    if mode == "line":
        draw = _draw_face_line
    elif mode == "points":
        draw = _draw_face_points
    elif mode == "numbers":
        draw = _draw_face_numbers
    else:
        raise ValueError("Invalid value for `mode` parameter.")
    for face in faces:
        draw(img_, face.dict, color=color, thick=thick)
    return img_


def draw_delauney(img, subdiv, *, color=(255, 0, 0), thick=1):
    """
    Draws a delauney triangulation based on cv2.Subdiv2D over facial landmarks.

    :param img: np.ndarray/cv2 image to draw over
    :param subdiv: cv2.Subdiv2D (result of phantom.align._delauney)
    :param color: tuple of BGR color to draw over `img`
    :return: np.ndarray (cv2 image) with the triangulation drawn over `img`
    """
    tri_list = subdiv.getTriangleList()
    ret = img.copy()
    for t in tri_list:
        p1, p2, p3 = tuple(t[0:2]), tuple(t[2:4]), tuple(t[4:6])
        cv2.line(ret, p1, p2, color, thick)  # , cv2.LINE_AA, 0)
        cv2.line(ret, p2, p3, color, thick)  # , cv2.LINE_AA, 0)
        cv2.line(ret, p3, p1, color, thick)  # , cv2.LINE_AA, 0)
    return ret


def imshow(img):
    """
    Renders an image with matplotlib.

    :param img: image to render.
    """
    # TODO: maybe extend to support rendering of multiple images?
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(temp)
    plt.show()
