r"""Provide console logging utilities using Rich for formatted
output."""

from __future__ import annotations

__all__ = ["log_char_diff", "log_markdown", "log_pretty"]

import logging
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty

logger: logging.Logger = logging.getLogger(__name__)


def log_markdown(msg: str, title: str | None = None) -> None:
    r"""Render a Markdown message to the console using Rich.

    Prints ``msg`` as rendered Markdown inside a titled panel using
    :class:`~rich.console.Console`. The ``level`` parameter is accepted
    for API compatibility but is currently unused.

    Args:
        msg: The message to render. May contain Markdown syntax.
        title: Optional title displayed in the panel border.

    Example:
        ```pycon
        >>> from glyphik.utils.logging import log_markdown
        >>> log_markdown("**hello**", title="Demo")

        ```
    """
    console = Console()
    console.print(Panel(Markdown(msg), title=title))


def log_pretty(data: Any, title: str | None = None) -> None:
    r"""Render a dictionary to the console in a pretty format using Rich.

    Prints ``data`` using :class:`~rich.pretty.Pretty` inside a titled
    panel via :class:`~rich.console.Console`. The ``level`` parameter is
    accepted for API compatibility but is currently unused.

    Args:
        data: The dictionary to render.
        title: Optional title displayed in the panel border.

    Example:
        ```pycon
        >>> from glyphik.utils.logging import log_pretty
        >>> log_pretty({"key": "value"}, title="Demo")

        ```
    """
    console = Console()
    console.print(Panel(Pretty(data), title=title))


def log_char_diff(before: str, after: str, *, label: str = "Text") -> None:
    """Log the character count before and after a text transformation
    step.

    Computes the signed difference and percentage change relative to the
    original length, then emits a single INFO log record. The sign of
    the diff reflects whether the transformation grew or shrank the
    text: negative means fewer characters, positive means more. Handles
    empty input gracefully (reports 0.0% change).

    Args:
        before: The text before transformation.
        after: The text after transformation.
        label: A short description of the step shown in the message,
            e.g. ``"HTML \u2192 Markdown"`` or ``"Markdown cleaning"``.
            Defaults to ``"Text"``.

    Example:
        ```pycon
        >>> from glyphik.utils.logging import log_char_diff
        >>> log_char_diff("<p>Hello</p>", "Hello", label="HTML \u2192 Markdown")

        ```
    """
    n_before = len(before)
    n_after = len(after)
    diff = n_after - n_before
    pct = diff / n_before * 100 if n_before > 0 else 0.0
    sign = "+" if diff >= 0 else ""
    logger.info(
        "%s: %s -> %s chars (%s%s chars, %s%.1f%%).",
        label,
        f"{n_before:,}",
        f"{n_after:,}",
        sign,
        f"{diff:,}",
        sign,
        abs(pct),
    )
