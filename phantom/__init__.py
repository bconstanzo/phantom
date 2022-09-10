import numpy as np  # so cv2 won't throw random errors if there's no prior install

import phantom.align
import phantom.enhance
import phantom.faces
import phantom.geometry
import phantom.measures
import phantom.similarity
import phantom.utils
import phantom.video
from importlib.metadata import version

__version__ = version("phantom")
