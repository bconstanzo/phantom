# still a work-in-progress
import numpy as np
import cv2


from phantom.utils import draw_faces
from phantom.faces import (
    encode, detect, estimate_age_onnx as estimate_age, estimate_gender_onnx as estimate_gender, age_tags,
)


video = cv2.VideoCapture(0)

def gender_name(x):
    if gender > 0.2: return "female"
    if gender < -0.2: return "male"
    return "not recognized"

while True:
    check, frame = video.read()
    faces = detect(frame, upsample=0)
    if faces:
        encodings = encode(frame, locations=faces)
        for i, e in enumerate(encodings):
            age = estimate_age(e)[0][0]
            age_text = age_tags[age]
            gender = estimate_gender(e)
            gender_text = gender_name(gender)
            left, top, right, bottom = faces[i]
            x = int (left * 0.5 + right * 0.5)
            y = top
            cv2.putText(frame, f"{age_text}", (x-19,y+21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(frame, f"{age_text}", (x-20,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.putText(frame, f"{gender_text}", (x-19,y+41), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.putText(frame, f"{gender_text}", (x-20,y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow("Caras", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
video.release()