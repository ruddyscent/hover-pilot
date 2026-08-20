"""Safe connectivity checks for first-time HoverPilot setup."""

from __future__ import annotations

import argparse
import socket
from typing import List, Optional

from hoverpilot.config import HOST, PORT


def check_tcp_connection(host: str, port: int, timeout_s: float) -> None:
    """Verify that the RFLink TCP endpoint accepts a connection."""
    with socket.create_connection((host, port), timeout=timeout_s):
        return


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether RealFlight Link is reachable without injecting a "
            "controller or sending flight controls."
        )
    )
    parser.add_argument("--host", default=HOST, help="RealFlight Link host")
    parser.add_argument("--port", type=int, default=PORT, help="RealFlight Link port")
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Connection timeout in seconds (default: 2).",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0.0:
        parser.error("--timeout must be greater than zero")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    print(f"[DOCTOR] Checking RealFlight Link at {args.host}:{args.port}...")
    try:
        check_tcp_connection(args.host, args.port, args.timeout)
    except OSError as exc:
        print(f"[DOCTOR] FAILED: {exc}")
        print(
            "[DOCTOR] Start RealFlight, enable RealFlight Link, and verify "
            "RFLINK_HOST/RFLINK_PORT from this machine."
        )
        return 1
    print("[DOCTOR] OK: TCP endpoint is reachable.")
    print("[DOCTOR] No controller was injected and no flight controls were sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
