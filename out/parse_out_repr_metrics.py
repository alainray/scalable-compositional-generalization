"""Compat wrapper for representation-metrics parser.

This keeps backwards compatibility with older command names while reusing the
implementation in parse_out_repr_epochs.py.
"""

from parse_out_repr_epochs import main


if __name__ == "__main__":
    main()
