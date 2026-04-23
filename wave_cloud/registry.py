"""
registry.py
-----------
Wave Function Registry — analogous to Google Cloud Functions registry.

Each Wave Function is a stateless, self-contained computation unit
that accepts a JSON payload and returns a JSON result.

Decorator usage (mirrors GCF):
    @wave_function("lyapunov")
    def fn_lyapunov(ctx: WaveContext) -> dict:
        ...
"""

import time
import traceback
from typing import Callable, Dict, Any
from .context import WaveContext


class FunctionRegistry:
    """
    Central registry for all Wave Functions.
    Handles dispatch, error handling, and execution timing.
    """

    def __init__(self):
        self._functions: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable):
        """Register a Wave Function by name."""
        self._functions[name] = fn
        print(f"[WaveCloud] Registered function: {name}")

    def wave_function(self, name: str):
        """
        Decorator to register a Wave Function.

        Example
        -------
        @registry.wave_function("my_func")
        def my_func(ctx: WaveContext) -> dict:
            x = ctx.get("x", 1.0)
            return {"result": x * 2}
        """
        def decorator(fn: Callable):
            self.register(name, fn)
            return fn
        return decorator

    def invoke(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a Wave Function by name.

        Parameters
        ----------
        name    : function name (str)
        payload : input JSON payload (dict)

        Returns
        -------
        result dict with keys:
            status  : "ok" | "error"
            data    : result payload
            elapsed : execution time [ms]
            function: name
        """
        if name not in self._functions:
            return {
                "status": "error",
                "function": name,
                "error": f"Wave Function '{name}' not found. "
                         f"Available: {list(self._functions.keys())}",
                "data": None,
                "elapsed": 0.0,
            }

        ctx = WaveContext(payload)
        t0  = time.perf_counter()
        try:
            result = self._functions[name](ctx)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return {
                "status":   "ok",
                "function": name,
                "data":     result,
                "elapsed":  round(elapsed, 3),
            }
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return {
                "status":   "error",
                "function": name,
                "error":    str(e),
                "traceback": traceback.format_exc(),
                "data":     None,
                "elapsed":  round(elapsed, 3),
            }

    def list_functions(self):
        """Return list of registered function names."""
        return list(self._functions.keys())

    def __repr__(self):
        return f"FunctionRegistry(functions={self.list_functions()})"


# ── Global registry singleton ────────────────────────────────────────
registry = FunctionRegistry()
wave_function = registry.wave_function
