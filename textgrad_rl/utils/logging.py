"""Small logging helpers."""

from __future__ import annotations

import platform
import sys
from typing import Any


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

