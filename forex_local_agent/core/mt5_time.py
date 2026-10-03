"""
MT5 Server-Time Helpers
=======================
MetaTrader 5 stamps deals, ticks and bars in *broker server time* encoded as
a Unix epoch (Tickmill: UTC+2 winter / UTC+3 summer). The Python API converts
any datetime passed to history_deals_get / copy_ticks_range into a real epoch
and compares it against those server-time stamps.

Consequence: a query like history_deals_get(now_utc - 2h, now_utc) silently
misses the most recent <offset> hours of deals. Every history query must go
through the helpers below.
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - MT5 missing only in CI-like envs
    mt5 = None

STATE_FILE = Path(__file__).resolve().parent.parent / "state" / "mt5_server_offset.json"
PROBE_SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD")
# Majors tick every few seconds while the market is open; an older tick can't be told
# apart from a different offset, so it is rejected.
MAX_TICK_AGE_SECONDS = 120

# The FX week opens Sunday and closes Friday at 17:00 New York time (needs the tzdata package on Windows)
NEW_YORK = ZoneInfo("America/New_York")
FX_WEEK_HOUR_NY = 17

_cached_offset: Optional[int] = None
_cached_at: float = 0.0
CACHE_TTL_SECONDS = 3600


def market_open(now_utc: datetime) -> bool:
    """Spot FX hours: Sunday 17:00 to Friday 17:00 New York time.

    That is 21:00 UTC during US daylight saving time and 22:00 UTC outside it
    (Malaysia: Monday 05:00/06:00 to Saturday 05:00/06:00); zoneinfo applies the shift.
    """
    ny = now_utc.astimezone(NEW_YORK)
    wd, h = ny.weekday(), ny.hour  # Mon=0 .. Sun=6
    if wd == 5:
        return False
    if wd == 6:
        return h >= FX_WEEK_HOUR_NY
    if wd == 4:
        return h < FX_WEEK_HOUR_NY
    return True


def next_market_open(now_utc: datetime) -> datetime:
    """Next Sunday 17:00 New York open, in UTC; `now_utc` itself if the market is already open."""
    if market_open(now_utc):
        return now_utc
    ny = now_utc.astimezone(NEW_YORK)
    sunday = (ny + timedelta(days=(6 - ny.weekday()) % 7)).date()
    open_ny = datetime(sunday.year, sunday.month, sunday.day, FX_WEEK_HOUR_NY, tzinfo=NEW_YORK)
    return open_ny.astimezone(timezone.utc)


def offset_from_tick_time(tick_epoch: int, now_epoch: float) -> Optional[int]:
    """Derive the server UTC offset (whole hours, in seconds) from a fresh tick.

    Returns None outside market hours or when the tick is too stale to be trusted.
    """
    if not market_open(datetime.fromtimestamp(now_epoch, tz=timezone.utc)):
        return None
    raw = tick_epoch - now_epoch
    rounded = int(round(raw / 3600.0) * 3600)
    if abs(raw - rounded) > MAX_TICK_AGE_SECONDS:
        return None
    if abs(rounded) > 14 * 3600:
        return None
    return rounded


def _load_state() -> Optional[int]:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return int(data["offset_seconds"])
    except Exception:
        return None


def _save_state(offset: int):
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps({"offset_seconds": offset, "measured_at": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug(f"Could not persist MT5 server offset: {e}")


def get_server_utc_offset_seconds(symbols: Iterable[str] = PROBE_SYMBOLS, force: bool = False) -> int:
    """Return broker server-time offset vs UTC in seconds (e.g. 10800 for UTC+3).

    Measured from the freshest live tick; falls back to the last persisted value
    when the market is closed, and to 0 (with a warning) if nothing is known.
    """
    global _cached_offset, _cached_at
    if not force and _cached_offset is not None and (time.time() - _cached_at) < CACHE_TTL_SECONDS:
        return _cached_offset

    measured = None
    if mt5 is not None:
        latest = 0
        for sym in symbols:
            try:
                tick = mt5.symbol_info_tick(sym)
                if tick and tick.time > latest:
                    latest = tick.time
            except Exception:
                continue
        if latest:
            measured = offset_from_tick_time(latest, time.time())

    if measured is not None:
        if measured != _load_state():
            _save_state(measured)
        _cached_offset, _cached_at = measured, time.time()
        return measured

    persisted = _load_state()
    if persisted is not None:
        _cached_offset, _cached_at = persisted, time.time()
        return persisted

    logger.warning("MT5 server UTC offset unknown (market closed, no cached value). Assuming 0.")
    return 0


def server_epoch_to_utc(server_epoch: int, offset: Optional[int] = None) -> datetime:
    """Convert an MT5 server-time epoch (deal.time, tick.time, bar time) to an aware UTC datetime."""
    off = get_server_utc_offset_seconds() if offset is None else offset
    return datetime.fromtimestamp(server_epoch - off, tz=timezone.utc)


def utc_to_server_query_dt(dt_utc: datetime, offset: Optional[int] = None) -> datetime:
    """Shift a real UTC datetime into the frame MT5 history queries compare against."""
    off = get_server_utc_offset_seconds() if offset is None else offset
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc + timedelta(seconds=off)


def history_deals_utc(start_utc: datetime, end_utc: datetime, offset: Optional[int] = None) -> List:
    """history_deals_get for a real-UTC window. Returns [] on failure."""
    if mt5 is None:
        return []
    off = get_server_utc_offset_seconds() if offset is None else offset
    deals = mt5.history_deals_get(utc_to_server_query_dt(start_utc, off), utc_to_server_query_dt(end_utc, off))
    return list(deals) if deals else []
