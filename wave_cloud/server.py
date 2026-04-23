"""
server.py
---------
Wave Cloud local server — FastAPI-based HTTP endpoint.
Mirrors Google Cloud Functions HTTP trigger interface.

Endpoints
---------
GET  /                         health check
GET  /functions                list registered Wave Functions
POST /run/{function_name}      invoke a Wave Function
POST /batch                    invoke multiple functions in sequence

Run:
    python -m wave_cloud.server --port 8080
    uvicorn wave_cloud.server:app --port 8080 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import uvicorn
import argparse

# Import registry and register all functions
from .registry  import registry
from . import functions   # side-effect: registers all Wave Functions

app = FastAPI(
    title="Wave Cloud",
    description="Serverless wave physics computing platform",
    version="1.0.0",
)


class InvokeRequest(BaseModel):
    payload: Dict[str, Any] = {}


class BatchRequest(BaseModel):
    jobs: List[Dict[str, Any]]
    # Each job: {"function": "lyapunov", "payload": {...}}


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "service": "Wave Cloud",
        "status":  "running",
        "version": "1.0.0",
        "functions_registered": len(registry.list_functions()),
    }


@app.get("/functions")
def list_functions():
    return {
        "functions": registry.list_functions(),
        "count": len(registry.list_functions()),
    }


@app.post("/run/{function_name}")
def run_function(function_name: str, req: InvokeRequest):
    result = registry.invoke(function_name, req.payload)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result)
    return JSONResponse(content=result)


@app.post("/batch")
def batch_run(req: BatchRequest):
    results = []
    for job in req.jobs:
        fn   = job.get("function", "")
        pl   = job.get("payload", {})
        res  = registry.invoke(fn, pl)
        results.append(res)
    return {"results": results, "count": len(results)}


# ── Entry point ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Wave Cloud local server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("wave_cloud.server:app",
                host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
