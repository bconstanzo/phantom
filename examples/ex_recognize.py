# still a work-in-progress
import numpy as np
import cv2


from phantom.utils import draw_faces
from phantom.faces import encode, landmark

known_faces = {
    "Bruno": "test/img4.jpg",
}

for name, impath in known_faces.items():
    img = cv2.imread(impath)
    if img is None:
        del(known_faces[name])
        continue
    known_faces[name] = encode(img)

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    faces = landmark(frame, upsample=1)
    frame = draw_faces(frame, faces)
    cv2.imshow("Caras", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()