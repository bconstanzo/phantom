import numpy as np
import cv2
import datetime
import phantom

from skl2onnx import to_onnx

from common import load_tagged, TaggedFace
from sklearn import svm


PATH_TAGGED = "tagged_faces.csv"


def load_data(tagged):
    """Build the (features, labels) matrices that sklearn expects.

    Faces are reduced to Phantom encodings so the model trains on compact
    embeddings instead of raw pixels, which keeps the regressor small and
    makes the exported ONNX input size independent of image resolution.
    """
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
    """Train the gender model and ship it as an ONNX file.

    ONNX is used so Phantom can run inference without depending on
    sklearn or on the Python version used for training. Per-stage timings
    are collected to spot which step dominates when the dataset grows.
    """
    print(f"Using phantom {phantom.__version__}")
    print("Loading tagged data...")
    t0 = datetime.datetime.now()
    tagged = load_tagged(PATH_TAGGED)
    print("Preparing data...")
    t1 = datetime.datetime.now()
    X, y = load_data(tagged)
    print("Training sklearn.svm.SVR...")
    t2 = datetime.datetime.now()
    reg = svm.SVR(kernel="linear", C=6.25, gamma=0.01)
    reg.fit(X, y)
    t3 = datetime.datetime.now()
    print("Done training!")
    print("Exporting to ONNX model...")
    n_features = X.shape[1]
    print(f"Tamaños de los inputs {n_features}")
    # to_onnx infers the input type from the sample X and picks a supported
    # opset by default, so no manual type/opset declaration is needed.
    # X is cast to float32 because that is what ONNX runtimes expect.
    onx = to_onnx(reg, X[:1].astype(np.float32))
    print("Creating ONNX model...")
    with open("gender_model.onnx.dat", "wb") as f:
        f.write(onx.SerializeToString())
    t4 = datetime.datetime.now()
    print("Done pickling!")
    print("Times:")
    print(f"\tLoading : {t1 - t0}")
    print(f"\tEncoding: {t2 - t1}")
    print(f"\tTraining: {t3 - t2}")
    print(f"\tExporting to ONNX: {t4 - t3}")


if __name__ == "__main__":
    main()