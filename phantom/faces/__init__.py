from .faces import (
    # first, the classes
    Shape, Shape5p, Shape68p, Face, Atlas,
    # and now the functions
    detect, landmark, encode, compare, estimate_gender
    # and not much more for now
)

from . import align