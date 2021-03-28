# still a work-in-progress
import numpy as np
import cv2


from phantom.utils import draw_faces
from phantom.faces import compare, encode, detect, estimate_age, age_tags

known_faces = {
    "Bruno": "c:/test/phantom/tests/img4.jpg",
    "Luciano": "c:/test/phantom/tests/luciano2.jpg",
    "Fer": "c:/test/phantom/tests/fer.jpg",
    "Ana": "c:/test/phantom/tests/ana.jpg",
    "Santi": "c:/test/phantom/tests/santi.jpg",
}

for name, impath in known_faces.items():
    img = cv2.imread(impath)
    if img is None:
        known_faces[name] = None
        continue
    known_faces[name] = encode(img)
    

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    faces = detect(frame, upsample=0)
    if faces:
        encodings = encode(frame, locations=faces)
        for i, e in enumerate(encodings):
            for k, v in known_faces.items():
                if compare(e, v) <= 0.6:
                    age = estimate_age(e)[0]
                    age_text = age_tags[age]
                    left, top, right, bottom = faces[i]
                    x = int (left * 0.5 + right * 0.5)
                    y = top
                    cv2.putText(frame, f"{k}", (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    cv2.putText(frame, f"{age_text}", (x-19,y+21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    cv2.putText(frame, f"{k}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.putText(frame, f"{age_text}", (x-20,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow("Caras", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()