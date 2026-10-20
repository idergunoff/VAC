"""Single application entry point."""
def main(argv=None):
    from .legacy import run
    return run(argv)
