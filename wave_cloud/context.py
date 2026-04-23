"""
context.py
----------
WaveContext — request context passed to every Wave Function.
Analogous to Flask's `request` object or GCF's `Request`.

Provides typed parameter access with defaults and validation.
"""

from typing import Any, Dict, Optional


class WaveContext:
    """
    Execution context for a Wave Function invocation.

    Parameters
    ----------
    payload : dict — raw input JSON payload

    Usage
    -----
    ctx = WaveContext({"w0": 1.0, "g1": 0.8, "T": 300})
    w0 = ctx.get("w0", default=1.0, cast=float)
    """

    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload or {}

    def get(self, key: str, default: Any = None, cast=None) -> Any:
        """
        Retrieve a parameter from the payload.

        Parameters
        ----------
        key     : parameter name
        default : default value if key not present
        cast    : type constructor (e.g. float, int, str)
        """
        val = self._payload.get(key, default)
        if cast is not None and val is not None:
            try:
                val = cast(val)
            except (ValueError, TypeError) as e:
                raise ValueError(f"WaveContext: cannot cast '{key}' to {cast}: {e}")
        return val

    def require(self, key: str, cast=None) -> Any:
        """Like get() but raises KeyError if key is missing."""
        if key not in self._payload:
            raise KeyError(f"WaveContext: required parameter '{key}' missing from payload")
        return self.get(key, cast=cast)

    def all(self) -> Dict[str, Any]:
        """Return entire payload dict."""
        return dict(self._payload)

    def __repr__(self):
        return f"WaveContext({self._payload})"
