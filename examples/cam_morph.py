import cv2
import numpy as np
import phantom


from phantom.faces.align import morph
from phantom.faces import detect, landmark
from phantom.utils import draw_faces, color_correct

video = cv2.VideoCapture(0)
newface = cv2.imread("c:/test/phantom/mask.jpg")
newface_marks = landmark(newface)

kern_erode = np.ones((7, 7), dtype=np.uint8)

def compose(img1, img2, mask):
    cv2.erode(mask, kern_erode, mask)
    mask = cv2.GaussianBlur(mask, (17, 17), 0)
    imask = -1. * (mask - 1.)
    ret = ((img1 * imask) + (img2 * mask)).astype(np.uint8)
    return ret

ploc_count = 1000

print(f"Running on phantom version {phantom.__version__}")

while True:
    check, frame = video.read()
    if ploc_count > 4:
        ploc = detect(frame, upsample=0)
        ploc_count = 0
    faces = landmark(frame, locations=ploc)
    ploc_count += 1
    # frame = draw_faces(frame, faces)
    if faces:
        morphed = morph(frame, newface, faces[0], newface_marks[0])
        faces_points = faces[0].points
        points = cv2.convexHull(np.int32(faces_points), returnPoints=False)
        mask = cv2.fillConvexPoly(
            np.zeros(frame.shape, dtype=np.float32),
            np.int32([faces_points[int(i)] for i in points]),
            (0.5, 0.5, 0.5))
        #m2 = (mask * morphed).astype(np.uint8)
        #morphed = color_correct(frame, morphed, faces[0])
    else:
        morphed = frame.copy()
        mask = np.zeros(frame.shape)
    out = compose(frame, morphed, mask)
    cv2.imshow("Morphing", out)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()