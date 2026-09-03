#!/usr/bin/env python3
"""Target-side smoke test for the Handstand Coach Yocto image."""

from pathlib import Path
from platform import machine, python_version


def main() -> None:
    message = (
        "Handstand Coach Yocto image is running "
        f"(architecture={machine()}, python={python_version()})"
    )
    print(message)
    Path("/run/handstand-hello.status").write_text(message + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
