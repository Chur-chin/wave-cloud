"""
wave_cloud
----------
Serverless wave physics computing platform.
Replaces Google Cloud Functions with wave-physics computation units.
"""

__version__ = "1.0.0"
__author__  = "Chur Chin"
__email__   = "tpotaoai@gmail.com"

from .registry import FunctionRegistry
from .context  import WaveContext

__all__ = ["FunctionRegistry", "WaveContext"]
