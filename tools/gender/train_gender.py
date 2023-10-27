import numpy as np
import cv2
import datetime
import phantom

from skl2onnx import convert_sklearn, get_latest_tested_opset_version
from skl2onnx.common.data_types import FloatTensorType

from common import load_tagged, TaggedFace
from sklearn import svm


PATH_TAGGED = "tagged_faces.csv"

def load_data(tagged):
    X = []
    y = []
    for t in tagged:
        enc = phantom.faces.encode(t.img)[0]
        X.append(enc)
        y.append(t.tag)
    X = np.array(X)
    y = np.array(y)
    return X, y


def main():
    print(f"Using phantom {phantom.__version__}")
    print("Loading tagged data...")
    t0 = datetime.datetime.now()
    tagged = load_tagged(PATH_TAGGED)
    print("Preparing data...")
    t1 = datetime.datetime.now()
    X, y = load_data(tagged)
    print("Training sklearn.svm.SVC...")
    t2 = datetime.datetime.now()
    reg = svm.SVR(kernel="linear", C=6.25, gamma=0.01)
    reg.fit(X, y)
    t3 = datetime.datetime.now()
    print("Done training!")
    print("Exporting to ONNX model...")
    target_opset = get_latest_tested_opset_version()
    n_features = X.shape[1]
    print(f'Tamaños de los inputs {n_features}')
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onx = convert_sklearn(
        reg, 
        initial_types=initial_type,
        target_opset={"": target_opset, "ai.onnx.ml": 1}
    )
    print("Creating ONNX model...")
    with open("gender_model.onnx.dat", "wb") as f:
        f.write(onx.SerializeToString())
    t4 = datetime.datetime.now()
    print("Done pickling!")
    print(f"Times:")
    print(f"\tLoading : {t1 - t0}")
    print(f"\tEncoding: {t2 - t1}")
    print(f"\tTraining: {t3 - t2}")
    print(f"\tExporting to ONNIX: {t4 - t3}")

if __name__ == "__main__":
    main()
