# phantom gender detector
This model is aimed at trying to detect the gender of a person, based on the
128-D face descriptors made by the dlib_face_recognition_resnet_model_v1
network.

The training dataset is derived from the dlib 5 point face landmark dataset
[1, 2].

Tools present in this directory:
  * `extract.py` walks over a directory and its images, and crops every face
    from them into a sepparate file, located in an output folder.
  * `tagger.py` is used to do manual tagging and training of "basic" SVM models.
  * `semisuper.py` is used to do semiautomatic tagging of larger datasets, and
    then presents an interface to review the (automatically) tagged data.

The first version of the gender estimation model was based around a single
script used to tag the images from the dataset (tagger.py). The second version
of the model has been thought out to address a few issues that the first model
had. To better handle the shortcommings, some improvements were made:

  * The v1 model had problems recognizing gender on small faces with few pixels
    in general. To address this problem, we added data augmentation, in
    particular the script downscales the training 5 times in 70% steps. This
    change, paired with a new tagging session, gave way to the v1.1 model.
  * The v1b model, while improved upon v1, had regressed specifically for the
    case of children, mislabeling faces that it previously labeled correctly.
  + `extract.py` and `semisuper.py` were developed accordingly to give better
    tools to work and evaluate on larger datasets easily.
  + The age estimation model was also trained, and used iteratively to improve
    the gender estimation model.

The tagger.py script does the following:

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
3.  If `FLAG_TEST` is set:
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
