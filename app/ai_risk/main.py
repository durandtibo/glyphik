"""
Date Range Picker — a Textual TUI app (text-input based)
Run with: python date_range_app.py
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from zoneinfo import available_timezones

import edgar
from edgar.entity.core import get_company

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll, Container
from textual.message import Message
from textual.reactive import reactive
from textual.validation import ValidationResult, Validator
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Select,
    SelectionList,
    Static,
    TabbedContent,
    TabPane,
)

# edgartools / SEC EDGAR requires a descriptive identity (name + email) on
# every request. Set the EDGAR_IDENTITY env var, e.g.
#   export EDGAR_IDENTITY="Your Name your.email@example.com"
# Falling back to a placeholder will still work for light use but SEC asks
# that you identify yourself properly.
edgar.set_identity(os.environ.get("EDGAR_IDENTITY", "Anonymous User anonymous@example.com"))


DATE_FORMAT = "YYYY-MM-DD"

# A curated, commonly-used set of time zones, with UTC pinned first.
_COMMON_TIMEZONES = [
    "UTC",
    "US/Eastern",
    "US/Central",
    "US/Mountain",
    "US/Pacific",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Sao_Paulo",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Europe/Moscow",
    "Africa/Cairo",
    "Africa/Johannesburg",
    "Asia/Dubai",
    "Asia/Kolkata",
    "Asia/Shanghai",
    "Asia/Tokyo",
    "Asia/Singapore",
    "Australia/Sydney",
    "Pacific/Auckland",
]
_ALL_TIMEZONES = set(available_timezones())
TIMEZONE_OPTIONS = [tz for tz in _COMMON_TIMEZONES if tz in _ALL_TIMEZONES or tz == "UTC"]
# Append any remaining IANA zones (alphabetical) not already listed, so every
# zone is selectable even though the dropdown leads with the common ones.
TIMEZONE_OPTIONS += sorted(_ALL_TIMEZONES - set(TIMEZONE_OPTIONS))


def parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


class IsValidDate(Validator):
    def validate(self, value: str) -> ValidationResult:
        if not value.strip():
            return self.success()  # empty is fine, just incomplete
        if parse_date(value) is None:
            return self.failure(f"Use {DATE_FORMAT}")
        return self.success()


def parse_identifier(value: str) -> tuple[str, str] | None:
    """Return (kind, normalized_value) where kind is 'ticker' or 'cik'."""
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return "cik", str(int(value))  # normalize away leading zeros
    if value.isalpha():
        return "ticker", value.upper()
    return None


class IsValidIdentifier(Validator):
    def validate(self, value: str) -> ValidationResult:
        if not value.strip():
            return self.success()
        if parse_identifier(value) is None:
            return self.failure("Enter a ticker (letters) or CIK (digits)")
        return self.success()


# --------------------------------------------------------------------------
# Selection panel — ticker/CIK, form types, date range, time zone, presets
# --------------------------------------------------------------------------
class SelectionPanel(VerticalScroll):
    """Self-contained widget for building a company filing search query."""

    DEFAULT_CSS = """
    SelectionPanel {
        width: 1fr;
        height: 100%;
        padding: 0 2 0 0;
    }

    SelectionPanel #title {
        text-style: bold;
        color: $accent;
        content-align: center middle;
        width: 100%;
        padding-bottom: 1;
    }

    SelectionPanel #range-display {
        border: round $secondary;
        padding: 1 2;
        margin: 1 0;
        background: $surface;
        color: $text;
        text-align: center;
    }

    SelectionPanel #inputs {
        height: auto;
        margin-bottom: 1;
    }

    SelectionPanel .field {
        width: 1fr;
        height: auto;
        margin: 0 1;
    }

    SelectionPanel .field Label {
        color: $text-muted;
        padding-bottom: 1;
    }

    SelectionPanel .field Input {
        border: round $primary-lighten-1;
    }

    SelectionPanel .field Input:focus {
        border: round $accent;
    }

    SelectionPanel #company-field {
        height: auto;
        margin-bottom: 1;
    }

    SelectionPanel #company-field Label {
        color: $text-muted;
        padding-bottom: 1;
    }

    SelectionPanel #company-field Input {
        border: round $primary-lighten-1;
    }

    SelectionPanel #company-field Input:focus {
        border: round $accent;
    }

    SelectionPanel #company-quick {
        height: auto;
        align: center middle;
        margin-bottom: 1;
    }

    SelectionPanel #company-quick Button {
        margin: 0 1;
        min-width: 10;
    }

    SelectionPanel #company-quick Button.-active {
        background: $accent;
        color: $background;
        text-style: bold;
    }

    SelectionPanel #forms-field {
        height: auto;
        margin-bottom: 1;
    }

    SelectionPanel #forms-field Label {
        color: $text-muted;
        padding-bottom: 1;
    }

    SelectionPanel #forms-select {
        height: auto;
        max-height: 6;
        border: round $primary-lighten-1;
        background: $surface;
    }

    SelectionPanel #forms-select:focus {
        border: round $accent;
    }

    SelectionPanel #tz-field {
        height: auto;
        margin-bottom: 1;
    }

    SelectionPanel #tz-field Label {
        color: $text-muted;
        padding-bottom: 1;
    }

    SelectionPanel #tz-field Select {
        border: round $primary-lighten-1;
    }

    SelectionPanel #presets {
        height: auto;
        align: center middle;
        margin-bottom: 1;
    }

    SelectionPanel #presets Button {
        margin: 0 1;
        min-width: 12;
    }

    SelectionPanel #actions {
        align: center middle;
        height: auto;
        margin-top: 1;
    }

    SelectionPanel #actions Button {
        margin: 0 1;
    }

    SelectionPanel #error-msg {
        color: $error;
        text-align: center;
        height: auto;
        display: none;
    }

    SelectionPanel #error-msg.-visible {
        display: block;
    }
    """

    start_date: reactive[date | None] = reactive(None)
    end_date: reactive[date | None] = reactive(None)
    timezone: reactive[str] = reactive("UTC")
    company: reactive[str] = reactive("")
    forms: reactive[tuple[str, ...]] = reactive(("10-K", "10-Q"))

    class Confirmed(Message):
        """Posted when the user confirms a valid selection."""

        def __init__(
            self,
            company: str,
            start_date: date,
            end_date: date,
            timezone: str,
            forms: tuple[str, ...],
        ) -> None:
            self.company = company
            self.start_date = start_date
            self.end_date = end_date
            self.timezone = timezone
            self.forms = forms
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._today = date.today()
        self._default_end = self._today
        self._default_start = self._today.replace(year=self._today.year - 1)

    def compose(self) -> ComposeResult:
        yield Label("🔍  Filing Search", id="title")

        with Vertical(id="company-field"):
            yield Label("Ticker or CIK")
            yield Input(
                placeholder="e.g. AAPL or 320193",
                id="company-input",
                validators=[IsValidIdentifier()],
            )

        with Horizontal(id="company-quick"):
            yield Button("AAPL", id="c-aapl", variant="primary")
            yield Button("NVDA", id="c-nvda", variant="primary")
            yield Button("MSFT", id="c-msft", variant="primary")

        with Vertical(id="forms-field"):
            yield Label("Form types")
            yield SelectionList[str](
                ("10-K", "10-K", True),
                ("10-Q", "10-Q", True),
                id="forms-select",
            )

        with Horizontal(id="inputs"):
            with Vertical(classes="field"):
                yield Label("Start date")
                yield Input(
                    placeholder=DATE_FORMAT,
                    id="start-input",
                    validators=[IsValidDate()],
                )
            with Vertical(classes="field"):
                yield Label("End date")
                yield Input(
                    placeholder=DATE_FORMAT,
                    id="end-input",
                    validators=[IsValidDate()],
                )

        yield Static("", id="error-msg")
        yield Static("Enter a start and end date above", id="range-display")

        with Vertical(id="tz-field"):
            yield Label("Time zone")
            yield Select(
                [(tz, tz) for tz in TIMEZONE_OPTIONS],
                value="UTC",
                id="tz-select",
                allow_blank=False,
            )

        with Horizontal(id="presets"):
            yield Button("Last 30d", id="p-30", variant="primary")
            yield Button("Last 90d", id="p-90", variant="primary")
            yield Button("Last year", id="p-year", variant="primary")
            yield Button("Clear", id="p-clear", variant="error")

        with Horizontal(id="actions"):
            yield Button("Confirm Selection", id="confirm", variant="success")

    def on_mount(self) -> None:
        self.set_inputs(self._default_start, self._default_end)
        self.query_one("#start-input", Input).focus()

    # -- helpers ----------------------------------------------------------
    def show_error(self, message: str) -> None:
        err = self.query_one("#error-msg", Static)
        if message:
            err.update(message)
            err.add_class("-visible")
        else:
            err.update("")
            err.remove_class("-visible")

    def update_display(self) -> None:
        disp = self.query_one("#range-display", Static)
        if self.start_date and self.end_date:
            if self.end_date < self.start_date:
                self.show_error("End date must be on or after the start date.")
                disp.update("Invalid range")
                return
            self.show_error("")
            days = (self.end_date - self.start_date).days + 1
            company_part = f"{self.company}  |  " if self.company else ""
            forms_part = f"  |  {', '.join(self.forms)}" if self.forms else "  |  no forms selected"
            disp.update(
                f"{company_part}{self.start_date.isoformat()}  →  {self.end_date.isoformat()}"
                f"   ({days} days, {self.timezone}){forms_part}"
            )
        else:
            self.show_error("")
            disp.update("Enter a start and end date above")

    def set_inputs(self, start: date, end: date) -> None:
        self.query_one("#start-input", Input).value = start.isoformat()
        self.query_one("#end-input", Input).value = end.isoformat()
        self.start_date = start
        self.end_date = end
        self.update_display()

    def set_company(self, value: str) -> None:
        self.query_one("#company-input", Input).value = value
        self.company = value
        for btn_id in ("c-aapl", "c-nvda", "c-msft"):
            btn = self.query_one(f"#{btn_id}", Button)
            btn.set_class(btn.label.plain == value, "-active")
        self.update_display()

    # -- events -------------------------------------------------------------
    def on_input_changed(self, event: Input.Changed) -> None:
        if not event.validation_result or event.validation_result.is_valid:
            if event.input.id == "start-input":
                self.start_date = parse_date(event.value)
                self.update_display()
            elif event.input.id == "end-input":
                self.end_date = parse_date(event.value)
                self.update_display()
            elif event.input.id == "company-input":
                parsed = parse_identifier(event.value)
                self.company = parsed[1] if parsed else event.value.strip().upper()
                for btn_id in ("c-aapl", "c-nvda", "c-msft"):
                    btn = self.query_one(f"#{btn_id}", Button)
                    btn.set_class(btn.label.plain == self.company, "-active")
                self.update_display()
        else:
            failures = event.validation_result.failure_descriptions
            self.show_error("; ".join(failures))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "tz-select" and event.value is not None:
            self.timezone = str(event.value)
            self.update_display()

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged) -> None:
        if event.selection_list.id == "forms-select":
            self.forms = tuple(event.selection_list.selected)
            self.update_display()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""

        if bid == "c-aapl":
            self.set_company("AAPL")
        elif bid == "c-nvda":
            self.set_company("NVDA")
        elif bid == "c-msft":
            self.set_company("MSFT")
        elif bid == "p-30":
            self.set_inputs(self._today - timedelta(days=29), self._today)
        elif bid == "p-90":
            self.set_inputs(self._today - timedelta(days=89), self._today)
        elif bid == "p-year":
            self.set_inputs(self._today.replace(year=self._today.year - 1), self._today)
        elif bid == "p-clear":
            self.query_one("#start-input", Input).value = ""
            self.query_one("#end-input", Input).value = ""
            self.start_date = None
            self.end_date = None
            self.update_display()
        elif bid == "confirm":
            if self.start_date and self.end_date and self.end_date >= self.start_date and self.forms:
                event.stop()
                self.post_message(
                    self.Confirmed(
                        company=self.company,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        timezone=self.timezone,
                        forms=self.forms,
                    )
                )
            elif not self.forms:
                self.app.notify("Please select at least one form type.", severity="warning")
            else:
                self.app.notify("Please enter a valid start and end date.", severity="warning")


# --------------------------------------------------------------------------
# Company info panel — fetches and displays a Company's rich representation
# --------------------------------------------------------------------------
class CompanyInfoPanel(VerticalScroll):
    """Self-contained widget that displays SEC EDGAR company info."""

    DEFAULT_CSS = """
    CompanyInfoPanel {
        width: 1fr;
        height: 1fr;
        border-left: solid $primary-lighten-1;
        padding: 0 0 0 2;
    }

    CompanyInfoPanel #company-title {
        text-style: bold;
        color: $accent;
        content-align: center middle;
        width: 100%;
        padding-bottom: 1;
    }

    CompanyInfoPanel #company-info {
        padding: 1 2;
        color: $text;
        border: round $secondary;
        background: $surface;
        height: 1fr;
    }
    """

    PLACEHOLDER = (
        "No company selected. Choose a ticker or CIK and click "
        "Confirm Selection to see company details here."
    )

    def compose(self) -> ComposeResult:
        yield Label("🏢  Company Info", id="company-title")
        yield Static(self.PLACEHOLDER, id="company-info")

    def show_placeholder(self) -> None:
        self.query_one("#company-info", Static).update(self.PLACEHOLDER)

    def load_company(self, identifier: str) -> None:
        """Fetch a company by ticker or CIK and render its __rich__ output."""
        info = self.query_one("#company-info", Static)

        if not identifier:
            self.show_placeholder()
            return

        info.update(f"Loading information for {identifier}…")

        try:
            lookup = int(identifier) if identifier.isdigit() else identifier
            company = get_company(lookup)
        except Exception as exc:  # noqa: BLE001 - surface any lookup/network error to the UI
            info.update(f"Could not load company info for {identifier}:\n{exc}")
            return

        info.update(company)  # Static renders Rich renderables, incl. Company.__rich__


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
class DateRangeApp(App):
    """A Textual app to pick a date range and view SEC company filings info."""

    CSS = """
    Screen {
        background: $background;
    }

    #root {
        width: 100%;
        height: 100%;
        padding: 1 2;
        background: $panel;
    }

    #tabs {
        width: 100%;
        height: 100%;
    }

    #tabs ContentSwitcher {
        height: 1fr;
    }

    TabPane {
        width: 100%;
        height: 100%;
    }

    #panels {
        width: 100%;
        height: 100%;
    }

    #right-column {
        width: 1fr;
        height: 100%;
    }

    #right-spacer {
        width: 100%;
        height: 1fr;
    }

    #analysis-placeholder {
        width: 100%;
        height: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="root"):
            with TabbedContent(id="tabs"):
                with TabPane("Search", id="tab-search"):
                    with Horizontal(id="panels"):
                        yield SelectionPanel(id="selection-panel")
                        with Vertical(id="right-column"):
                            yield CompanyInfoPanel(id="company-panel")
                            yield Static(id="right-spacer")
                with TabPane("Analysis", id="tab-analysis"):
                    yield Static("Nothing here yet.", id="analysis-placeholder")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "textual-dark"

    def on_selection_panel_confirmed(self, event: SelectionPanel.Confirmed) -> None:
        company_part = f"{event.company} — " if event.company else ""
        self.notify(
            f"{company_part}Range confirmed: {event.start_date} → {event.end_date} "
            f"({event.timezone}) — Forms: {', '.join(event.forms)}",
            title="Date Range",
            severity="information",
        )
        self.query_one(CompanyInfoPanel).load_company(event.company)


if __name__ == "__main__":
    DateRangeApp().run()