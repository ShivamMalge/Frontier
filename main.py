# main.py

"""Repo-root shim for the CLI, which lives in ``app/cli.py``.

Installing the project gives you a ``frontier`` command that calls the same
function; this file only exists so ``python main.py`` keeps working in a
checkout, without an install.
"""

from app.cli import run

if __name__ == "__main__":
    run()
