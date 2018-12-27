# phantom
a library for working with faces in python

## Requirements
* Python 3.6.x
* NumPy 1.15+
* matplotlib
* scikit-learn 0.20+
* OpenCV 4+
* dlib 19.8.1

## Installing on Windows
Since Windows tends to be a bit harder to get things working on, we have
compiled this instructions to get you going:

1. Get latest Python 3.6.x.
2. Install Numpy, OpenCV and scikit-learn from Christoph Gohlkes wheels 
https://www.lfd.uci.edu/~gohlke/pythonlibs/.
3. We've compiled dlib 19.16.0 with AVX for Python 3.6 on Windows 64 bits. You
   can find it in the /tools directory. Compared to non-AVX version, it's about
   2.5 times faster.

    Dlib is developed and maintained by Davis E. King. Check www.dlib.net and
    https://github.com/davisking/dlib.

4. You canstill download the wheel for dlib 19.8.1 from
   https://pypi.org/simple/dlib/ (thanks to 
   https://github.com/charlielito/install-dlib-python-windows for the pointer).
   __We don't recommend this, and will eventually drop support for older
   versions of dlib__.

Linux and macOS should be easier. If you have problems, point us at it so that
we can work out detailed instructions.

## Notes
* When usign our compiled dlib wheel, you may get a puzzling ImportError whein
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
