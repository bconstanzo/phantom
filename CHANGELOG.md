# Changelog
# 1.0
(once a stable API is reached)

## 0.3
- Add model training directory in the repo:
    - tagger tool, used for phantom_gender_model.
- New models.
    - OpenCVs DNN facial detector.
    - Gender predictor based on facial encodings (own).

## 0.2
* Tooling for datasets and training.
* Code leanup and changes:
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
