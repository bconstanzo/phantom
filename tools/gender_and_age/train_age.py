import numpy as np
import cv2
import datetime
import phantom

from skl2onnx import to_onnx

from common import load_tagged, TaggedFace
from sklearn import svm, metrics


PATH_TAGGED = "tagged_faces.csv"
PATH_MODEL = "age_model.onnx.dat"


def load_data(tagged):
    """Build the (features, labels) matrices that sklearn expects.

    Faces are reduced to Phantom encodings so the classifier trains on
    compact embeddings instead of raw pixels, and the exported ONNX input
    size does not depend on image resolution.
    """
    X = []
    y = []
    for t in tagged:
        enc = phantom.faces.encode(t.img)[0]
        X.append(enc)
        y.append(t.age_tag)
    X = np.array(X)
    y = np.array(y)
    return X, y


def main():
    """Train the age classifier and export it as an ONNX file.

    ONNX lets Phantom run inference without sklearn or the training-time
    Python version. Returns the model and data so they can be inspected
    from an interactive session (`ipython -i train_age.py`), e.g. to
    compare sklearn vs ONNX predictions.
    """
    print(f"Using phantom {phantom.__version__}")
    print("Loading tagged data...")
    t0 = datetime.datetime.now()
    tagged = load_tagged(PATH_TAGGED)
    print("Preparing data...")
    t1 = datetime.datetime.now()
    X, y = load_data(tagged)
    print("Training sklearn.svm.SVC...")
    t2 = datetime.datetime.now()
    clf = svm.SVC(kernel="linear", C=6.25, gamma=0.01)
    # parameters were explored on a grid search, particularle C was
    # picked to get a good balance in a validation split of 0.3 for the
    # tagged data:
    #     from sklearn.model_selection import train_test_split
    #     # ...
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    # HOWEVER for training we now use the full tagged data
    clf.fit(X, y)
    t3 = datetime.datetime.now()
    print("Done training!")
    print("Exporting to ONNX model...")
    # zipmap=False makes the scores a plain tensor instead of a list of
    # dicts, which is far easier to consume from onnxruntime.
    # X is cast to float32 because that is what ONNX runtimes expect.
    onx = to_onnx(
        clf,
        X[:1].astype(np.float32),
        options={id(clf): {"zipmap": False}},
    )
    with open(PATH_MODEL, "wb") as f:
        f.write(onx.SerializeToString())
    t4 = datetime.datetime.now()
    print("Done exporting!")
    print("Times:")
    print(f"    Loading : {t1 - t0}")
    print(f"    Encoding: {t2 - t1}")
    print(f"    Training: {t3 - t2}")
    print(f"    Exporting to ONNX: {t4 - t3}")
    return clf, X, y


if __name__ == "__main__":
    clf, X_train, y_train = main()