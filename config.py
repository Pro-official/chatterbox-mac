import os
from typing import Optional, Tuple

# Environment-based configuration
# Production (default): AUDIOBOOK_DEV_MODE unset or "false"
# Development: AUDIOBOOK_DEV_MODE=true
DEV_MODE = os.getenv("AUDIOBOOK_DEV_MODE", "false").lower() == "true"

# Dev mode limits
DEV_MAX_CHUNKS = 1      # Only 1 chapter
DEV_MAX_WORDS = 1000    # Max words per chapter


def get_test_mode_limits(test_mode: bool) -> Tuple[Optional[int], Optional[int]]:
    """Return (max_chunks, max_words) based on test_mode flag."""
    if test_mode:
        return DEV_MAX_CHUNKS, DEV_MAX_WORDS
    return None, None
