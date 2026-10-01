"""
Bubat AI Autonomous Skills Registry
===================================
Exports quantitative and analytical skills for use across the trading agent ecosystem.
"""

from .economic_calendar_filter import (
    EconomicCalendarFilter,
    calendar_filter,
    is_trade_permitted_by_calendar,
    get_economic_events_summary
)
from .currency_correlation_filter import (
    CurrencyCorrelationFilter,
    decompose_pair,
    is_trade_permitted_by_correlation,
    get_currency_exposure
)

__all__ = [
    "EconomicCalendarFilter",
    "calendar_filter",
    "is_trade_permitted_by_calendar",
    "get_economic_events_summary",
    "CurrencyCorrelationFilter",
    "decompose_pair",
    "is_trade_permitted_by_correlation",
    "get_currency_exposure",
]
