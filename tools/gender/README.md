# phantom gender detector
This model is aimed at trying to detect the gender of a person, based on the
128-D face descriptores made by the dlib_face_recognition_resnet_model_v1
network.

The training dataset is derived from the dlib 5 point face landmark dataset
[1, 2]. The tagger.py script does the following:

1.  If `FLAG_TAG` is set:
    1.  Open the `path` directory, and then each image inside it.
    2.  For each image, locate the face, encode it and draw the landmarks.
    3.  Display the image with the facial landmarks drawn over the face.
    4.  A count of how many faces are in each set ("male" or "female") is drawn
        over the drawn images, so that the user can decide to skip faces to
        balance the dataset.
    5.  Wait for the user to tag (hit key "M" or "F") or skip ("SPACE" key).
    6.  Repeat 2, until user hits "Q" to finish.
    7.  If `FLAG_SAVE` is set, the dataset is saved to a pickle file,
        for each tagged image saving:
        *  image path
        *  face location
        *  face landmarks
        *  128-D encoding
        *  gender
2.  If `FLAG_TRAIN` is set:
    1.  If tagging was skipped, load the tagging data from the pickle file.
    2.  An RBF-kernel SVM dlib object is created, configured and trained [3].
    3.  The SVM is pickled and saved.
3. If `FLAG_TEST` is set:
    1.  The SVM is loaded (from disk, or already present in memory).
    2.  The thest dataset is loaded (a list of dictionarys with path and gender
        keys.
    3.  The faces are located, encoded and compared with the ground truth value
        for each test image. Results are counted.
    4.  The results are shown, and if `FLAG_TEST_SAVE` is set, saved to a report
        file.

### References
1.  https://github.com/davisking/dlib-models
2.  http://dlib.net/files/data/dlib_faces_5points.tar
3.  http://dlib.net/svm_binary_classifier.py.html
