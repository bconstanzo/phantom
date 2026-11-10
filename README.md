# phantom
forensic image processing in python

## Requirements
* Python 3.14.x (3.11.x+ still supported, but we recommend you use the latest version)
* NumPy 1.15+
* scipy 1.2.1+
* matplotlib
* scikit-learn 1.0+
* OpenCV 4.5+
* dlib 19.23+ (19.16 and 19.8 still work)
* ONNX 1.12+
* protobuf 3.16.0 (as per ONNX runtime dependency)

## Installing on Windows, Linux and MacOS

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