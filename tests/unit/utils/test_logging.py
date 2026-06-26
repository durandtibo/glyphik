from __future__ import annotations

from unittest.mock import patch

from glyphik.utils.logging import log_char_diff, log_markdown, log_pretty

MODULE = "glyphik.utils.logging"


##################################
#     Tests for log_markdown     #
##################################


def test_log_markdown_with_rich() -> None:
    log_markdown("# Hello")


def test_log_markdown_with_rich_with_title() -> None:
    log_markdown("# Hello", title="cats")


def test_log_markdown_returns_none() -> None:
    assert log_markdown("# Hello") is None


def test_log_markdown_empty_string() -> None:
    log_markdown("")


################################
#     Tests for log_pretty     #
################################


def test_log_pretty_with_rich() -> None:
    log_pretty({"hello": "world"})


def test_log_pretty_with_rich_with_title() -> None:
    log_pretty({"hello": "world"}, title="cats")


def test_log_pretty_returns_none() -> None:
    assert log_pretty({"hello": "world"}) is None


def test_log_pretty_empty_dict() -> None:
    log_pretty({})


##################################
#   Tests for log_char_diff      #
##################################


# --- Calls logger.info once ---


def test_log_char_diff_calls_info_once() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_char_diff("before", "after")
    mock_logger.info.assert_called_once()


# --- Label ---


def test_log_char_diff_default_label_appears_in_message() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_char_diff("abc", "a")
    assert "Text" in str(mock_logger.info.call_args)


def test_log_char_diff_custom_label_appears_in_message() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_char_diff("abc", "a", label="HTML → Markdown")
    assert "HTML → Markdown" in str(mock_logger.info.call_args)


# --- Reduction (after < before) ---


def test_log_char_diff_reduction_message_contains_before_count() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_char_diff("a" * 1000, "a" * 200)
    assert "1,000" in str(mock_logger.info.call_args)


def test_log_char_diff_reduction_message_contains_after_count() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_char_diff("a" * 1000, "a" * 200)
    assert "200" in str(mock_logger.info.call_args)


def test_log_char_diff_reduction_diff_is_negative() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_char_diff("a" * 1000, "a" * 200)
    assert "-800" in str(mock_logger.info.call_args)


# --- Increase (after > before) ---


def test_log_char_diff_increase_diff_is_positive() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_char_diff("a" * 100, "a" * 150)
    args = mock_logger.info.call_args.args
    assert "+" in args
    assert "50" in args


# --- No change ---


def test_log_char_diff_no_change_diff_is_zero() -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_char_diff("abc", "abc")
    args = mock_logger.info.call_args.args
    assert "+" in args
    assert "0" in args


# --- Empty input ---


def test_log_char_diff_empty_before_does_not_raise() -> None:
    log_char_diff("", "abc")


def test_log_char_diff_both_empty_does_not_raise() -> None:
    log_char_diff("", "")
