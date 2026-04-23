#!/usr/bin/env python3
"""
wavectl.py
----------
Wave Cloud CLI — analogous to `gcloud` for Google Cloud.

Usage examples
--------------
# List available Wave Functions
python wavectl.py functions

# Run a Wave Function
python wavectl.py run lyapunov --w0 1.0 --g1 0.8 --T 300
python wavectl.py run synapse --theta 45 --intensity 5e5
python wavectl.py run polariton --band I --d_nm 10
python wavectl.py run stdp --dt_min -80 --dt_max 80
python wavectl.py run hbn_epsilon --omega_min 600 --omega_max 1700
python wavectl.py run bifurcation --w0 1.0 --n_points 50
python wavectl.py run scaling_law --n_w0 6

# Run via HTTP server (must be running separately)
python wavectl.py run lyapunov --remote --host localhost --port 8080
"""

import argparse
import json
import sys


def run_local(function_name: str, payload: dict) -> dict:
    """Invoke Wave Function directly (no server needed)."""
    from wave_cloud.registry import registry
    from wave_cloud import functions   # register all
    return registry.invoke(function_name, payload)


def run_remote(function_name: str, payload: dict,
               host: str = "localhost", port: int = 8080) -> dict:
    """Invoke Wave Function via HTTP server."""
    import requests
    url = f"http://{host}:{port}/run/{function_name}"
    resp = requests.post(url, json={"payload": payload})
    return resp.json()


def print_result(result: dict, pretty: bool = True):
    if pretty:
        status = result.get("status", "?")
        fn     = result.get("function", "?")
        elapsed= result.get("elapsed", 0)
        print(f"\n{'='*55}")
        print(f"  Wave Function : {fn}")
        print(f"  Status        : {status}")
        print(f"  Elapsed       : {elapsed} ms")
        print(f"{'='*55}")
        data = result.get("data", result.get("error", {}))
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_payload(args_list: list) -> dict:
    """
    Parse --key value pairs from remaining CLI args into a dict.
    Attempts float/int conversion; falls back to string.
    """
    payload = {}
    i = 0
    while i < len(args_list):
        if args_list[i].startswith("--"):
            key = args_list[i][2:]
            if i + 1 < len(args_list) and not args_list[i+1].startswith("--"):
                raw = args_list[i+1]
                try:
                    val = int(raw)
                except ValueError:
                    try:
                        val = float(raw)
                    except ValueError:
                        val = raw
                payload[key] = val
                i += 2
            else:
                payload[key] = True
                i += 1
        else:
            i += 1
    return payload


def main():
    parser = argparse.ArgumentParser(
        prog="wavectl",
        description="Wave Cloud CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wavectl functions
  wavectl run lyapunov --w0 1.0 --g1 0.8
  wavectl run synapse --theta 45 --intensity 500000
  wavectl run polariton --band I --d_nm 10
  wavectl run stdp
  wavectl run hbn_epsilon
  wavectl run bifurcation --w0 1.0 --n_points 40
  wavectl run scaling_law --n_w0 6
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- functions ---
    subparsers.add_parser("functions", help="List registered Wave Functions")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run a Wave Function")
    run_parser.add_argument("function", help="Wave Function name")
    run_parser.add_argument("--remote",  action="store_true",
                            help="Use HTTP server instead of local execution")
    run_parser.add_argument("--host",    default="localhost")
    run_parser.add_argument("--port",    type=int, default=8080)
    run_parser.add_argument("--json",    action="store_true",
                            help="Output raw JSON")

    args, extra = parser.parse_known_args()

    if args.command == "functions":
        from wave_cloud.registry import registry
        from wave_cloud import functions
        fns = registry.list_functions()
        print("\nRegistered Wave Functions:")
        for f in fns:
            print(f"  • {f}")
        print(f"\nTotal: {len(fns)}")

    elif args.command == "run":
        payload = parse_payload(extra)
        if args.remote:
            result = run_remote(args.function, payload, args.host, args.port)
        else:
            result = run_local(args.function, payload)
        print_result(result, pretty=not args.json)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
