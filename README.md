# phantom
forensic image processing in python

## Requirements
* Python 3.10.x (3.6.x+ still supported, but recommend latest)
* NumPy 1.15+
* NumPy 1.22+ (per ONNX runtime)
* scipy 1.2.1+
* matplotlib
* scikit-learn 1.0+
* OpenCV 4.5+
* dlib 19.23+ (19.16 and 19.8 still work)
* ONNX 1.12+
* protobuf 3.16.0 (as per ONNX runtime dependency)

## Installing on Windows
Since Windows tends to be a bit harder to get things working on, we have
compiled this instructions to get you going:

1. Get latest Python 3.10.x or 3.11.x (both have been tested and are supported).
2. Install Numpy, Scipy, OpenCV, scikit-learn, and matplotlib (in that order)
   from Christoph Gohlkes wheels at https://www.lfd.uci.edu/~gohlke/pythonlibs/.
   * Note: these wheels are no longer available. We recomend the same approach
     as Linux and macOS: `pip install -r requirements.txt`, but consider that
     those builds may not be as optimized as Gohlkes wheels were. Things may
     run a bit slower. If you have available old optimized wheels, prefer those
     to speed things up (quite) a bit.
3. We've compiled dlib 19.16.0, 19.23.0, and 19.24.2 with AVX for Python 3.6,
   3.10, and 3.11 on Windows 64 bits. You can find the corresponding wheels
   in the /tools directory. Compared to non-AVX version, these are about 2.5
   times faster.

    Dlib is developed and maintained by Davis E. King. Check www.dlib.net and
    https://github.com/davisking/dlib.

1. You can still download the wheel for dlib 19.8.1 from
4. You can still download the wheel for dlib 19.8.1 from
   https://pypi.org/simple/dlib/ (thanks to 
   https://github.com/charlielito/install-dlib-python-windows for the pointer).
   __We don't recommend this, and will eventually drop support for older
   versions of dlib__.

Linux and macOS are simpler:

1. Open a console and go to the directory where you've cloned phantom.
2. Run `pip install -r requirements.txt`.
3. Run `pip install .`.

You'll need (and most probably already have installed) the platform compiler and
it will build a the dependencies as it installs them. This method also works on
Windows, but it's less probable that you'll have a proper build environment
already set up, and you'll end up with a lower performance installation.

## Notes
* When usign our compiled dlib wheel, you may get a puzzling ImportError when
  trying to import dlib (not phantom). If this happens, you can fix it by
  installing Intel MKL (download from https://software.intel.com/en-us/mkl) and
  adding the following values to your environment variables:
  
  * Add to PATH the following values:

    `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\tbb\vc_mt;`

    `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\mkl;`

    `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\compiler;`

  * LIB (you may have to create it):

    `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb\lib\intel64_win\vc_mt;`

    `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\lib\intel64_win;`

    `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\compiler\lib\intel64_win;`

  * INCLUDE
    
    `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\include;`

    `C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\include\intel64\ilp64;`
