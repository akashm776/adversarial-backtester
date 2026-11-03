"""Compatibility entry point: install the package, then run `python scaffold.py`."""

from adversarial_backtester.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
