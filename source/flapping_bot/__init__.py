"""Top-level package marker for the flapping_bot extension.

This allows reliable imports like `import flapping_bot.flapping_bot` in
environments where namespace packages (PEP 420) may not be recognized by the
import machinery. It re-exports the inner package for convenience.
"""

from .flapping_bot import *  # noqa: F401,F403

