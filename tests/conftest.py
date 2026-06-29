from __future__ import annotations

import os

import pytest

from glyphik.utils.imports import is_edgar_available


@pytest.fixture(scope="session", autouse=True)
def _edgar_identity() -> None:
    """Set SEC EDGAR identity once for the entire test session.

    Reads the identity string from the ``EDGAR_IDENTITY`` environment
    variable and passes it to :func:`edgar.set_identity`.  SEC EDGAR
    requires a name and email address in the format
    ``"First Last name@example.com"`` so that they can contact you if
    your usage violates their policies.

    Skipped silently when ``edgar`` is not installed.

    Raises:
        ValueError: If ``edgar`` is installed but ``EDGAR_IDENTITY``
            is not set.
    """
    if not is_edgar_available():
        return

    import edgar

    identity = os.getenv("EDGAR_IDENTITY")
    if not identity:
        msg = (
            "EDGAR_IDENTITY environment variable is not set. "
            "Set it to your name and email, e.g. "
            "'First Last name@example.com'."
        )
        raise ValueError(msg)
    edgar.set_identity(identity)
