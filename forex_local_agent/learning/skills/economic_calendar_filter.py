"""
Economic Calendar Filter Skill - Bubat AI
==========================================
Provides real-time high-impact macroeconomic calendar filtering and news shock protection.
Prevents the agent from entering positions right before high-impact economic releases (CPI, NFP, Interest Rates)
where broker spreads widen 5x-10x and directional slippage causes catastrophic stop-outs.
"""

import os
import json
import httpx
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger


class EconomicCalendarFilter:
    """
    Fetches, caches, and evaluates high-impact economic calendar events.
    """

    FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    def __init__(self, cache_ttl_seconds: int = 7200):
        self.cache_ttl = cache_ttl_seconds
        self.cache_file = Path(__file__).resolve().parent / "calendar_cache.json"
        self._cached_events: List[Dict[str, Any]] = []
        self._last_fetch_time: Optional[datetime] = None
        self._load_cache()

    def _load_cache(self):
        """Load cached calendar events from disk if valid."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._cached_events = data.get("events", [])
                    fetch_iso = data.get("last_fetch")
                    if fetch_iso:
                        self._last_fetch_time = datetime.fromisoformat(fetch_iso)
            except Exception as e:
                logger.debug(f"[CalendarFilter] Failed to load cache: {e}")

    def _save_cache(self):
        """Save events to local cache file."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "last_fetch": datetime.now(timezone.utc).isoformat(),
                    "events": self._cached_events
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"[CalendarFilter] Failed to save cache: {e}")

    def refresh_calendar(self, force: bool = False) -> List[Dict[str, Any]]:
        """Fetch latest weekly economic events from ForexFactory JSON feed."""
        now = datetime.now(timezone.utc)
        if not force and self._last_fetch_time and (now - self._last_fetch_time).total_seconds() < self.cache_ttl:
            return self._cached_events

        try:
            with httpx.Client(timeout=6.0, headers={"User-Agent": "BubatAI/1.0"}) as client:
                resp = client.get(self.FEED_URL)
                if resp.status_code == 200:
                    events = resp.json()
                    self._cached_events = events
                    self._last_fetch_time = now
                    self._save_cache()
                    logger.info(f"[CalendarFilter] Refreshed {len(events)} calendar events from feed.")
                    return self._cached_events
        except Exception as e:
            logger.warning(f"[CalendarFilter] Failed to fetch live calendar ({e}). Using existing cache.")

        return self._cached_events

    def extract_currencies(self, symbol: str) -> List[str]:
        """Extract base and quote currencies from symbol (e.g., EURUSD -> ['EUR', 'USD'])."""
        sym = symbol.upper()
        if sym == "XAUUSD":
            return ["USD"]
        if len(sym) == 6:
            return [sym[:3], sym[3:]]
        # Crosses or non-standard
        currencies = []
        for c in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]:
            if c in sym:
                currencies.append(c)
        return currencies or ["USD"]

    def _parse_event_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse ISO datetime with timezone offset."""
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def get_upcoming_events(self, symbol: str, window_hours: int = 12) -> List[Dict[str, Any]]:
        """
        Get all upcoming High and Medium impact events for a given currency pair within window_hours.
        """
        self.refresh_calendar()
        currencies = self.extract_currencies(symbol)
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(hours=window_hours)

        relevant_events = []
        for ev in self._cached_events:
            country = ev.get("country", "").upper()
            impact = ev.get("impact", "")
            if country in currencies and impact in ("High", "Medium"):
                ev_time = self._parse_event_datetime(ev.get("date", ""))
                if ev_time and now - timedelta(minutes=15) <= ev_time <= horizon:
                    diff_mins = int((ev_time - now).total_seconds() / 60)
                    relevant_events.append({
                        "country": country,
                        "title": ev.get("title", ""),
                        "impact": impact,
                        "time_utc": ev_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "minutes_away": diff_mins,
                        "forecast": ev.get("forecast", ""),
                        "previous": ev.get("previous", ""),
                    })

        relevant_events.sort(key=lambda x: x["minutes_away"])
        return relevant_events

    def is_trade_permitted_by_calendar(
        self,
        symbol: str,
        blackout_before_mins: int = 30,
        blackout_after_mins: int = 15
    ) -> Tuple[bool, str]:
        """
        Check if an entry is blocked due to an imminent or freshly released High-Impact event.
        
        Args:
            symbol: Forex pair (e.g. 'EURUSD')
            blackout_before_mins: Minutes before release to stop new entries (default 30m)
            blackout_after_mins: Minutes after release before spreads normalize (default 15m)
            
        Returns:
            (permitted: bool, reason: str)
        """
        self.refresh_calendar()
        currencies = self.extract_currencies(symbol)
        now = datetime.now(timezone.utc)

        for ev in self._cached_events:
            country = ev.get("country", "").upper()
            impact = ev.get("impact", "")
            if country in currencies and impact == "High":
                ev_time = self._parse_event_datetime(ev.get("date", ""))
                if not ev_time:
                    continue

                diff_seconds = (ev_time - now).total_seconds()
                diff_minutes = diff_seconds / 60.0

                # 1. Before release blackout window
                if 0 <= diff_minutes <= blackout_before_mins:
                    title = ev.get("title", "High-Impact Release")
                    msg = f"BLACKOUT: [{country}] '{title}' release in {int(diff_minutes)}m"
                    return False, msg

                # 2. Immediately after release blackout window (spread blowout cushion)
                if -blackout_after_mins <= diff_minutes < 0:
                    title = ev.get("title", "High-Impact Release")
                    mins_ago = abs(int(diff_minutes))
                    msg = f"BLACKOUT: [{country}] '{title}' released {mins_ago}m ago (volatility buffer)"
                    return False, msg

        return True, "CLEAR"

    def format_calendar_summary(self, symbol: str) -> str:
        """Format a human-readable and LLM-friendly summary of upcoming events."""
        events = self.get_upcoming_events(symbol, window_hours=12)
        if not events:
            return "No high/medium impact calendar events scheduled in next 12h."

        lines = []
        for ev in events[:4]:
            timing = f"in {ev['minutes_away']}m" if ev['minutes_away'] > 0 else f"{abs(ev['minutes_away'])}m ago"
            lines.append(f"[{ev['country']}] {ev['title']} ({ev['impact']} Impact, {timing})")

        return " | ".join(lines)


# Singleton instance for ecosystem export
calendar_filter = EconomicCalendarFilter()


def is_trade_permitted_by_calendar(symbol: str, blackout_before: int = 30, blackout_after: int = 15, **kwargs) -> Tuple[bool, str]:
    """Helper functional API to check trade permission against live calendar."""
    before = kwargs.get("pre_buffer_mins", blackout_before)
    after = kwargs.get("post_buffer_mins", blackout_after)
    return calendar_filter.is_trade_permitted_by_calendar(symbol, before, after)


def get_economic_events_summary(symbol: str) -> str:
    """Helper functional API to get prompt-ready event string."""
    return calendar_filter.format_calendar_summary(symbol)
