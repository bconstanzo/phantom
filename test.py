# relevar_entorno.py — pensado para Python 3.11, sin dependencias extra.
# Uso: python relevar_entorno.py > entorno_py311.txt
import importlib.metadata as md
import platform
import sys

print("== Sistema")
print("python  :", sys.version.replace("\n", " "))
print("platform:", platform.platform(), "|", platform.machine())

print("\n== Paquetes relevantes")
relevantes = ("numpy", "scipy", "matplotlib", "scikit-learn", "opencv",
              "dlib", "onnx", "onnxruntime", "protobuf", "setuptools", "pytest", "phantom")
for dist in sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower()):
    nombre = dist.metadata["Name"]
    if any(r in nombre.lower() for r in relevantes):
        print(f"{nombre}=={dist.version}")

print("\n== OpenCV")
try:
    import numpy as np  # phantom importa numpy antes que cv2
    import cv2
    print("cv2.__version__      :", cv2.__version__)
    print("cv2.__file__         :", cv2.__file__)
    print("contrib (xfeatures2d):", hasattr(cv2, "xfeatures2d"))
    print("threads / optimized  :", cv2.getNumThreads(), "/", cv2.useOptimized())
    claves = ("Version control", "C++ standard", "CPU/HW features", "Baseline",
              "Dispatched", "Parallel framework", "Lapack", "Eigen", "IPP", "numpy")
    for linea in cv2.getBuildInformation().splitlines():
        if any(k in linea for k in claves):
            print("  " + linea.strip())
except Exception as e:
    print("error importando cv2:", repr(e))

print("\n== dlib")
try:
    import dlib
    print("dlib.__version__:", dlib.__version__)
    print("DLIB_USE_CUDA   :", getattr(dlib, "DLIB_USE_CUDA", "n/d"))
except Exception as e:
    print("error importando dlib:", repr(e))

print("\n== pip freeze completo")
for dist in sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower()):
    print(f"{dist.metadata['Name']}=={dist.version}")