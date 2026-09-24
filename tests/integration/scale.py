"""Scale knob shared across the integration suite.

`RAGCORE_IT_SCALE` defaults to "full" — the whole 8-book synthetic library,
per the instruction not to shortcut coverage for speed. Set it to "quick"
for fast local iteration (a 3-book subset, fewer generated queries); CI and
any "does this actually hold at scale" question should always run "full".
"""

from __future__ import annotations

import os


def scale() -> str:
    return os.environ.get("RAGCORE_IT_SCALE", "full")


def is_quick() -> bool:
    return scale() == "quick"
