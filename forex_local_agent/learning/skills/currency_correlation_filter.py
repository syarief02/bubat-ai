"""
Currency Correlation & Exposure Filter Skill
=============================================
Part of the Bubat AI Autonomous Quantitative Toolset.

Prevents portfolio over-concentration and correlation risk by:
1. Decomposing currency pairs into Base and Quote currencies.
2. Tracking gross exposure per currency (e.g., max 3 active positions involving JPY).
3. Tracking net directional exposure (e.g., preventing stacking 3+ Short JPY or Long USD positions).
4. Providing real-time correlation protection to prevent correlated multi-stop cascades.
"""

from typing import List, Dict, Any, Tuple, Optional
from loguru import logger

# Major and minor currencies tracked
KNOWN_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "XAU", "XAG"}

# Default maximum number of open positions allowed that contain any single currency
DEFAULT_MAX_CURRENCY_EXPOSURE = 3


def decompose_pair(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Decomposes a forex symbol into (base_currency, quote_currency).
    Strips broker suffixes such as .raw, .ecn, _m, +, etc.
    """
    if not symbol or not isinstance(symbol, str):
        return None, None

    clean = symbol.upper().replace(".RAW", "").replace(".ECN", "").replace("+", "").replace("-", "").strip()

    # Check for known 3-char base currency prefix
    for curr in KNOWN_CURRENCIES:
        if clean.startswith(curr):
            base = curr
            quote = clean[len(curr):len(curr) + 3]
            if quote in KNOWN_CURRENCIES:
                return base, quote

    # Fallback: Standard 6-character forex pairs (e.g. EURUSD)
    if len(clean) >= 6:
        base = clean[:3]
        quote = clean[3:6]
        return base, quote

    return None, None


class CurrencyCorrelationFilter:
    """
    Evaluates active positions across the account to prevent correlated currency risk.
    """

    def __init__(self, max_currency_exposure: int = DEFAULT_MAX_CURRENCY_EXPOSURE):
        self.max_currency_exposure = max_currency_exposure

    def calculate_portfolio_exposure(self, open_positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """
        Calculates total count and net directional exposure for each currency.

        Returns:
            Dict mapping currency -> {
                "total": int (gross number of open trades involving this currency),
                "long": int (number of trades where this currency is bought),
                "short": int (number of trades where this currency is sold),
                "net": int (long - short)
            }
        """
        exposure: Dict[str, Dict[str, int]] = {}

        for pos in open_positions:
            sym = pos.get("symbol", "")
            base, quote = decompose_pair(sym)
            if not base or not quote:
                continue

            # In MT5: type 0 = BUY, type 1 = SELL
            ptype = pos.get("type")
            is_buy = ptype == 0 or str(pos.get("action", "")).upper() == "BUY"

            # Base currency: bought if BUY, sold if SELL
            base_dir = "long" if is_buy else "short"
            # Quote currency: sold if BUY, bought if SELL
            quote_dir = "short" if is_buy else "long"

            for curr, direction in [(base, base_dir), (quote, quote_dir)]:
                if curr not in exposure:
                    exposure[curr] = {"total": 0, "long": 0, "short": 0, "net": 0}
                exposure[curr]["total"] += 1
                exposure[curr][direction] += 1
                exposure[curr]["net"] = exposure[curr]["long"] - exposure[curr]["short"]

        return exposure

    def is_trade_permitted(
        self,
        symbol: str,
        action: str,
        open_positions: List[Dict[str, Any]],
        max_currency_exposure: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Determines whether a proposed trade on `symbol` with `action` violates
        currency concentration or directional stacking limits.

        Args:
            symbol: Target trade symbol (e.g. 'GBPJPY')
            action: 'BUY' or 'SELL'
            open_positions: List of open position dicts from MT5
            max_currency_exposure: Optional override for max positions per currency

        Returns:
            Tuple of (permitted: bool, reason: str)
        """
        max_exposure = max_currency_exposure or self.max_currency_exposure
        base, quote = decompose_pair(symbol)
        if not base or not quote:
            return True, "Unknown symbol format, bypass correlation check."

        action_upper = action.upper().strip()
        is_buy = action_upper == "BUY"
        base_dir = "long" if is_buy else "short"
        quote_dir = "short" if is_buy else "long"

        exposure = self.calculate_portfolio_exposure(open_positions)

        # Check Base Currency
        base_exp = exposure.get(base, {"total": 0, "long": 0, "short": 0, "net": 0})
        if base_exp["total"] >= max_exposure:
            return False, f"Maximum currency exposure reached for {base} ({base_exp['total']}/{max_exposure} active positions)"

        # Check Quote Currency
        quote_exp = exposure.get(quote, {"total": 0, "long": 0, "short": 0, "net": 0})
        if quote_exp["total"] >= max_exposure:
            return False, f"Maximum currency exposure reached for {quote} ({quote_exp['total']}/{max_exposure} active positions)"

        # Directional Stacking Check: Block if already holding max_exposure in the EXACT same direction
        if base_exp.get(base_dir, 0) >= max_exposure:
            return False, f"Maximum directional exposure reached for {base} ({base_dir.upper()} count: {base_exp[base_dir]}/{max_exposure})"

        if quote_exp.get(quote_dir, 0) >= max_exposure:
            return False, f"Maximum directional exposure reached for {quote} ({quote_dir.upper()} count: {quote_exp[quote_dir]}/{max_exposure})"

        return True, f"Exposure verified: {base} total={base_exp['total']}, {quote} total={quote_exp['total']}"


# Global default instance
_default_filter = CurrencyCorrelationFilter()


def is_trade_permitted_by_correlation(
    symbol: str,
    action: str,
    open_positions: List[Dict[str, Any]],
    max_currency_exposure: int = DEFAULT_MAX_CURRENCY_EXPOSURE
) -> Tuple[bool, str]:
    """Top-level convenience function for integration into risk engines."""
    return _default_filter.is_trade_permitted(
        symbol=symbol,
        action=action,
        open_positions=open_positions,
        max_currency_exposure=max_currency_exposure
    )


def get_currency_exposure(open_positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Top-level convenience function to get active currency exposure breakdown."""
    return _default_filter.calculate_portfolio_exposure(open_positions)
