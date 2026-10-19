"""Model failures whose local, controlled messages may be shown to the user.

Only construct ProviderError with messages defined by this application. Never
pass through provider response bodies, exception strings, prompts or credentials.
Unexpected exceptions remain private in the HTTP boundary.
"""
from __future__ import annotations

import re


class DocumentError(ValueError):
    """A local, controlled document validation message safe for the API."""


class ProviderError(ValueError):
    """Distinguish an expected model failure from an unknown third-party error."""

    status_code = 502

    def __init__(self, code: str, public_message: str):
        # Keep the diagnostic code safe for structured logs, even if a caller
        # accidentally supplies something other than a local category name.
        self.code = code if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) else "provider_error"
        self.public_message = public_message
        super().__init__(public_message)
