import cv2
import datetime
import numpy as np
import phantom
import onnxruntime as rt
import numpy as np

from common import load_tagged, TaggedFace
from sklearn import svm, metrics


PATH_TAGGED = "tagged_faces.csv"

# instead of having a main() function we run the code as-is so that we can
# access the variables from running it with python -i
print(f"Using phantom {phantom.__version__}")
print("Loading tagged data...")
t0 = datetime.datetime.now()
tagged = load_tagged(PATH_TAGGED)
print("Preparing data...")
t1 = datetime.datetime.now()
total = 0
wrong_age = 0
 
print("Import ONNX model...")
model = rt.InferenceSession("age_model.onnx", providers=['CPUExecutionProvider'])
input_name = model.get_inputs()[0].name
print("Starting test...")

for t in tagged:
    enc = phantom.faces.encode(t.img)[0]
    t.enc = enc
    tag = t.tag
    age_tag = t.age_tag
    #print(enc.dtype)
    age, _ = model.run(None, {input_name: [enc.astype(np.float32)]})

    #age = phantom.faces.estimate_age(enc)
    if age != age_tag:
        wrong_age += 1
    total += 1
print("Done testing!")
print(f"    Total:{total:18}")
print(f"    Acc. age:{100*(total - wrong_age)/total:15}")

print(f"Times:")
print(f"    Loading : {t1 - t0}")






