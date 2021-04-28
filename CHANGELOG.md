# Changelog
# 1.0
(once a stable API is reached)

## 0.8.0
* Improvements to face clustering. `faces.Atlas` now includes `.group()` and
  `.predict()` that work together to group all the faces in an atlas (using
  DBSCAN clustering with reasonable defaults) and then you can match new faces
  against those clusters (done through a Nearest Centroid Classifier, built from 
  the groups found beforehand).
    * This implementations are going to be improved upon in following versions.
      Currently, they work for small-to-medium datasets (a couple hundred 
      people over thousands of images), but they lose accuracy over larger
      datasets (thousands of people over tens of thousands of images).
    * Moving forward, a different implementation of facial encodings might be
      necessary to overcome the current limitations.
* New feature: face landmark normalization that can be used as a helper for
  different tasks.
  * There's no example script yet, but you can see `tools/emotions` for details.
* In the `faces` module, `detect()`, `landmark()` and `encode()` functions now
  transparently convert images from BGR to RGB to run the corresponding models.
  Previously this had to be done by hand, and when not performed it'd lead to
  a slight accuracy loss in the models (particularly the encoding model suffered
  from this).
  * This issue arises from the fact that OpenCV stores in memory the images in
    BGR order and most other software assumes RGB. If were doing the conversion
    by hand, you can use the set the keyword-only flag `bgr=False` when calling
    or you can remove the conversion entirely and let the functions handle it.
    Speed and memory-wise the impact of doing the conversions is small.
* Making `phantom-models` a separate PyPI project is still in the plans, but the
  fact that we're considering changing the dlib models we're currently using for
  other implementations means there will be changes. For the moment 0.8.0 is a
  GitHub only release.

## 0.7.3
* Restores the Shape68p model and fixes a few bugs.

## 0.7.2
* Uploaded to PyPI.
  * Removed the 68-point facial shape landmark model. It's coming back once we
    get approved for larger project size at PyPI. You can download it from 
    `tools`, manually place it on the `phantom/models` directory and then
    patching `phantom/faces/faces.py` to load the correct model.

  * Fixed issues with 0.7.1, which was meant to be the first PyPI version.

## 0.7
* Changes to models:
  * Gender estimation model has been refined to cover a few edge cases and a
    data augmentation approach has been taken.
    * Since this model has been trained on par with the age estimation model,
      performance is now generally better across all age ranges, though
      particular cases have regressed.
  * An age estimation model has been trained that works over facial encodings.
  * `tools/gender/tagger.py` has been improved to support additional tags other
    than gender (age and emotion have been applied for the new models).

    New scripts have been added to `tools/gender`, to refactor the growing
    `tagger.py` and move functionality out of it into separate, concise scripts.
    There's work under way to further improve the current models.
* Explored but not implemented (yet): object search via image descriptors.
  * Didn't pan out since the best available descriptor is SIFT, but it's
    patented. Check `phantom/tools/descriptors` for details.
  * Alternatives to SIFT are available, and there could be potentially other
    ways to do descriptor based search.

## 0.6
* Keep reorganizing features:
    * Functionality in the `align` module moved on to the `faces` module,
      since its intent was the alignment of faces with respect to one another.
* Re-structuring of modules:
    * `align` is now empty, but will have algorithms and functions related to
      aligning images with respect to other images, instead of face-specific
      alignment functions as it used to.
    * `enhance` has image enhancement functions.
    * `faces` gathers all the face-related functionality (detecting, landmarking
      encoding, aligning, etc). `phantom.faces` is now a full sub module (not 
      just a file).
    * `geometry` gathers the geometric transforms and associated functions.
    * `measure` will have functions related to calibrating and measuring objects
      in images.
    * `similarity` has (image) perceptual hashing functions.
    * `utils` gathers all utility functions that aren't core functionality of
      a module.
    * `video` has functionality that eases working over videos.
* Lazy loading of models used in the `faces` module.
    * While the solution works, there's hope for a better mechanism when we move
      to Python 3.7+ (in the future).

## 0.5.5
* June 2019 Workshop freeze.
* Reorganized the project for more general use cases.
    + Added `enhance` module, dedicated to image enhancement operations.
    * Added `geometry` module, dedicated to geometric transformations.
    * * Added `similarity` module, dedicated to similarity metrics.
      * We've tested the implementations against imagehash, and have tried to
        be as compatible with them as we could. That said, the scaling
        algorithms from OpenCV don't behave the same as those from PIL and there
        are small differences that result in our implementation not being
        compatible with imagehash -- simply because the same image will give a
        different hash depending on which library you use.

## 0.5
+ Move part of the clustering logic from the example into the `faces.Atlas`
  class. This clears the example code and helps to build new features on top of
  clustering (and the resulting clusters).
* Moved on to dlib 19.16. (still remain 19.8.1 compatible)

## 0.4
* Added scikit-learn as a requirement.
    * The cluster example has been adapted to use DBSCAN and KMeans as
      clustering algorithms (from scikit-learn). Combined, they give better
      accuracy when detecting the number of people and, and then classifying
      them into (more) correct labels. The KMeans object also has a .predict()`
      method that helps matching new faces against a known dataset.

## 0.3
* Added /tools directory in the repo. This directory holds tools and files
  used to create custom models:
    * tools/gender contains tagger.py, a tool used to create the training
      dataset for phantom_gender_model, based on dlibs 5 point face landmark
      dataset.
* New models:
    * Gender predictor based on facial encodings (own). Implemented in 
      `faces.estimate_gender`.
* Changed requirements to OpenCV 4.

## 0.2
* Tooling for datasets and training.
* Code cleanup and changes:
    * Changes to how `faces.Shape` and subclasses work.
        * `faces.Shape` subclasses now know the model with which they work and
          how to draw their points/lines over faces.
    * Fixed `utils.draw_faces` (and related functions) to work with Shape
      objects.
    * Deleted old, un-used functions.
* New examples and changes:
    * Examples were renamed for better clarity. Those named with "cam_" at the
      beginning require a webcam.
    * examples/cam_recognize.py finds a known face in webcam input
    * examples/cluster.py walks over a path, finding faces on images, then
      encodes them and applies clustering to detect how many people are present
      in them.
* setup.py

## 0.1
* Published to GitHub.
* Files/directories cleanup.

### 0.0.3
* `faces` module implemented.
* `align.morph` implemented.

### 0.0.2
* Started implementing `align.morph`.
* Added stub for `faces` module.
* Reorganized a few files on the repository.

### 0.0.1
* `align` and `utils` modules implemented.
