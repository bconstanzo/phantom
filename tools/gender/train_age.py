import cv2
import datetime
import numpy as np
import phantom
import pickle


from common import load_tagged, TaggedFace
from sklearn import svm, metrics


PATH_TAGGED = "tagged_faces.csv"


def main():
    print(f"Using phantom {phantom.__version__}")
    print("Loading tagged data...")
    t0 = datetime.datetime.now()
    tagged = load_tagged(PATH_TAGGED)
    print("Preparing data...")
    t1 = datetime.datetime.now()
    X = []
    y = []
    for t in tagged:
        enc = phantom.faces.encode(t.img)[0]
        X.append(enc)
        y.append(t.age_tag)
    X = np.array(X)
    y = np.array(y)
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
    print("Pickling...")
    with open("age_model.pickle", "wb") as handle:
        pickle.dump(clf, handle)
    t4 = datetime.datetime.now()
    print("Done pickling!")
    print(f"Times:")
    print(f"    Loading : {t1 - t0}")
    print(f"    Encoding: {t2 - t1}")
    print(f"    Training: {t3 - t2}")
    print(f"    Pickling: {t4 - t3}")

if __name__ == "__main__":
    main()
