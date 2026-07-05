from __future__ import annotations

import pytest

from glyphik.prompts.summarization import GENERIC_SYSTEM_PROMPT


@pytest.mark.parametrize("prompt", [GENERIC_SYSTEM_PROMPT])
def test_verify_prompts_are_string(prompt: str) -> None:
    assert isinstance(prompt, str)
