import datetime
import numpy as np
import phantom
import onnxruntime as rt
from common import load_tagged


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
wrong_gender = 0
 
print("Import ONNX model...")
model = rt.InferenceSession("gender_model.onnx", providers=['CPUExecutionProvider'])
input_name = model.get_inputs()[0].name
print("Starting test...")

for t in tagged:
    enc = phantom.faces.encode(t.img)[0]
    t.enc = enc
    tag = t.tag
    gender, _ = model.run(None, {input_name: [enc.astype(np.float32)]})
    if gender != tag:
        wrong_gender += 1
    total += 1
print("Done testing!")
print(f"    Total:{total:18}")
print(f"    Acc. gender:{100*(total - wrong_gender)/total:15}")
print(f"Times:")
print(f"    Loading : {t1 - t0}")