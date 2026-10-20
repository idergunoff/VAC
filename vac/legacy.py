"""Compatibility entry points; state now belongs to Controller instances."""
from .app import create_window, main


def run(argv=None):
    return main(argv)
