"""
Various algorithms to align faces.
"""

import cv2
import numpy as np


def perspective(base, dest, baselm, destlm, *, inter=cv2.INTER_CUBIC):
    """
    Aligns `dst` with respect to `base`, calculating an homography between them
    and using a perspective transformation (cv2.warpPerspective).

    :param base: base image, taken as reference frame
    :param dest: image to be modified
    :param baselm: `phantom.faces.Shape` object describing a face in base
    :param destlm: `phantom.faces.Shape` object describing a face in dest
    :param inter: interpolation algorithm from cv2 (see cv2.INTER_ constants)
    :return: `dst`, warped to match the `base` image (according to `baselm` and
        `destlm`)
    """
    base_points = np.float32(baselm.points)
    dest_points = np.float32(destlm.points)
    # now we calculate the homography, and return dst warped
    h, _status = cv2.findHomography(dest_points, base_points)
    size = (base.shape[1], base.shape[0])
    return cv2.warpPerspective(dest, h, size, flags=inter)


def procrustes(base, dest, baselm, destlm, *, inter=cv2.INTER_CUBIC):
    """
    Aligns `dst` with respect to `base`, using Ordinary Procrustes Analysis (see
    https://matthewearl.github.io/2015/07/28/switching-eds-with-python/).

    :param base: base image, taken as reference frame
    :param dst: image to be modified
    :param baselm: `phantom.faces.Shape` object describing a face in base
    :param destlm: `phantom.faces.Shape` object describing a face in dest
    :param inter: interpolation algorithm from cv2 (see cv2.INTER_ constants)
    :return: `dst`, warped to match the `base` image (according to `baselm` and
        `destlm`)
    """
    # prepare the landmark points
    base_points = np.matrix(baselm.points, dtype=np.float32)
    dest_points = np.matrix(destlm.points, dtype=np.float32)
    # now we follow Matthews instructions
    # this is roughly similar to his transform_from_points function
    c1 = np.mean(base_points, axis=0)
    c2 = np.mean(dest_points, axis=0)
    base_points -= c1
    dest_points -= c2
    s1 = np.std(base_points)
    s2 = np.std(dest_points)
    base_points /= s1
    dest_points /= s2
    U, _S, Vt = np.linalg.svd(base_points.T * dest_points, full_matrices=False)
    R = (U * Vt).T
    M = np.vstack(
        [np.hstack(((s2 / s1) * R,
         c2.T - (s2 / s1) * R * c1.T)),
         np.matrix([0., 0., 1.])
         ])
    # and now from the warp_im function
    size = (base.shape[1], base.shape[0])
    return cv2.warpAffine(dest,
                   M[:2],
                   size,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP | inter)


def _borders(img):
    """
    Helper function for `morph`.

    Returns the border points for an image. Used by `morph` to extend the points
    before delauney triangulation is applied.

    :param img: image from which the border points are computed
    :return: list of points that make the border of the image

    Computes 12 points (4 on each side), approximately in this fashion:
        X..X..X..X
        ..........
        X........X
        ..........
        X........X
        ..........
        X..X..X..X
    """
    # TODO: extend the algorithm to accept a "number of points" parameter so it won't be a fixed
    #   set of points.
    s = img.shape
    x0, x1, x2, x3 = map(lambda x: int(s[1] * x), [0, 0.33, 0.67, 1])
    y0, y1, y2, y3 = map(lambda x: int(s[0] * x), [0, 0.33, 0.67, 1])
    x3 -= 1
    y3 -= 1
    return [
        (x0, y0), (x1, y0), (x2, y0), (x3, y0),
        (x0, y1),                     (x3, y1),
        (x0, y2),                     (x3, y2),
        (x0, y3), (x1, y3), (x2, y3), (x3, y3),
    ]


def _delauney(img, points):
    """
    Helper function for `morph`.

    Performs Delauney Triangulation on a set of points.

    :param img: np.ndarray/cv2 image to subdivide
    :param points: points to subdivide the image
    :return: cv2.Subdiv2D object, already subdivided
    """
    rect = (0, 0, img.shape[1], img.shape[0])
    div = cv2.Subdiv2D(rect)
    for p in points:
        div.insert(p)
    return div


def _make_tri(tri):
    """
    Helper function for `morph`.

    Transforms a triangle from cv2.Subdiv2D.getTriangleList() (flat 6 element
    array) into a 3-row 2D-point array.

    :param tri: an element of cv2.Subdiv2D.getTriangleList().
    :return: np.float32, 3x2 array.
    """
    return np.float32([
        tri[0:2],
        tri[2:4],
        tri[4:6],
    ])


def morph(base, dest, baselm, destlm, *, inter=cv2.INTER_CUBIC, add_points=False):
    """
    Morphs `dest` to match `base`, according to the facial landmarks passed.

    :param base: base image, taken as reference frame
    :param dest: image to be modified
    :param baselm: `phantom.faces.Shape` object describing a face in base
    :param destlm: `phantom.faces.Shape` object describing a face in dest
    :param inter: interpolation algorithm from cv2 (see cv2.INTER_ constants)
    :return: `dst`: morphed to match the `base` image (according to `baselm` and
        `destlm`)
    """
    def translate_delauney(tri, points1, points2):
        """
        Helps filter the `base_triangle` for a correct morphing.
        """
        out = []
        for p in tri:
            idx = points1.index((int(p[0]), int(p[1])))
            out.extend(points2[idx])
        return _make_tri(out)

    # prepare the list of points
    base_points = baselm.points
    dest_points = destlm.points
    if add_points:
        base_points += _borders(base)
        dest_points += _borders(dest)
    # delauney triangulation
    base_div = _delauney(base, base_points)
    dest_div = _delauney(dest, dest_points)
    base_triangles = base_div.getTriangleList()
    dest_triangles = dest_div.getTriangleList()
    # we have to clean the triangles list because cv2.Subdiv2D is a bit aggressive...
    # # bmax = max(base.shape)
    # # base_triangles = [_make_tri(t) for t in base_triangles if all(0 <= e < bmax for e in t)]
    # # dest_triangles = [translate_delauney(t, base_points, dest_points) for t in base_triangles]
    dmax = max(dest.shape)
    dest_triangles = [_make_tri(t) for t in dest_triangles if all(0 <= e < dmax for e in t)]
    base_triangles = [translate_delauney(t, dest_points, base_points) for t in dest_triangles]
    ret = np.zeros(base.shape)
    for s_tri, d_tri in zip(dest_triangles, base_triangles):
        # heres where morphing begins
        # first we calculate the bounding rectangles for each triangle
        x, y, w, h = cv2.boundingRect(s_tri)
        s_roi = dest[y: y + h, x: x + w]
        s_tri -= np.float32([x, y])
        x, y, w, h = cv2.boundingRect(d_tri)
        d_roi = ret[y: y + h, x: x + w]
        d_tri -= np.float32([x, y])
        # ...we generate the triangular mask and its inverse
        mask = np.zeros((h, w, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(d_tri), (1.0, 1.0, 1.0))  # , cv2.LINE_AA, 0)
        imask = -1. * (mask - 1)
        # and now we warp and "render" on the target image
        warp_mat = cv2.getAffineTransform(s_tri, d_tri)
        render = cv2.warpAffine(s_roi, warp_mat, (w, h), None, flags=inter)  # , borderMode=cv2.BORDER_DEFAULT)
        ret[y: y + h, x: x + w] = (render * mask) + (d_roi * imask)
    # TODO: maybe we could work something on the order of rendering?
    return ret.astype(np.uint8)
