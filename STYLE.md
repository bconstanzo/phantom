# Code Style Guide for phantom
The guiding principle behind the code style we have chosen is clarity and 
readability. We're wrapping some confusing and (for some) difficult to use
libraries, best we can do is being straightforward about it.

It bears reminding:

> This is just a guide

That said, here are the rules:

* Variable names are lower case, with underscores. Most times, names should
  have meaning, though comprehensions and variables in short iterations may be
  forgiven if they don't.
    * `face`, `img`, `warp_mat` are ok names.
    * `known_face`, `face_locations` are better.
    * `x` and `y` are acceptable if in a comprehension: `[x + y for x, y in
      points]`. If this isn't the case, refactor for clarity.
* Classes names are CamelCase/PascalCase, without underscores. Meaning
  is essential, but   you should try to keep the names short. Usually classes
  are within a module or submodule, and that should give enough context to
  understand a short name.
* Functions follow the same rules as variable names. Usually a function name
  is a verb: `encode`, `detect`, `draw`. Some functions have special
  conventions, for example, the `align` module has a few functions that are
  named after the algorithm they implement.
* Line length:
    * 80 chars hard for documentation (markdown files) and docstrings.
    * 100 chars soft for code, 120 chars hard.
        * If you reach 100 lines, try to break them. If you reach 120, you
          *must* break it
