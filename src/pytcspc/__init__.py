from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytcspc")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .presets import *
from .sdt import *
from .spc import *
from .util import *
from .flim import *
from .fcs import *
from .batch import *