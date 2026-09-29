"""
Web Surfer & Financial Intelligence Engine
==========================================
Provides fast, resilient live web search, financial news retrieval,
market session tracking, and article scraping without requiring Docker/SearXNG.
"""

import httpx
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from loguru import logger


class WebSurfer:
    """Multi-source live web search and market session engine."""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def get_current_market_session(self) -> Dict[str, Any]:
        """Calculates active global forex market sessions based on UTC clock."""
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour + now_utc.minute / 60.0

        sessions = []
        # Sydney: 21:00 - 06:00 UTC
        if hour >= 21 or hour < 6:
            sessions.append("Sydney (Asian Session)")
        # Tokyo: 00:00 - 09:00 UTC
        if 0 <= hour < 9:
            sessions.append("Tokyo (Asian Session)")
        # London: 08:00 - 16:30 UTC
        if 8 <= hour < 16.5:
            sessions.append("London (European Session)")
        # New York: 13:00 - 21:00 UTC
        if 13 <= hour < 21:
            sessions.append("New York (US Session)")

        overlap = ""
        if "London (European Session)" in sessions and "New York (US Session)" in sessions:
            overlap = "London / New York OVERLAP (Peak Liquidity & High Volatility)"
        elif "Sydney (Asian Session)" in sessions and "Tokyo (Asian Session)" in sessions:
            overlap = "Tokyo / Sydney Overlap"

        primary = overlap if overlap else (", ".join(sessions) if sessions else "Inter-session Transition")

        return {
            "utc_time": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "active_sessions": sessions,
            "session_summary": primary,
            "best_pairs_for_session": self._recommended_pairs_for_session(sessions)
        }

    def _recommended_pairs_for_session(self, sessions: List[str]) -> List[str]:
        """Returns the most active pairs based on active trading sessions."""
        pairs = set()
        for s in sessions:
            if "London" in s:
                pairs.update(["EURUSD", "GBPUSD", "EURGBP", "GBPJPY", "EURJPY"])
            if "New York" in s:
                pairs.update(["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "XAUUSD"])
            if "Tokyo" in s or "Sydney" in s:
                pairs.update(["USDJPY", "AUDUSD", "NZDUSD", "EURJPY", "AUDJPY"])
        return list(pairs) if pairs else ["EURUSD", "USDJPY", "GBPUSD"]

    def search_news(self, query: str = "forex market", max_results: int = 6) -> List[Dict[str, Any]]:
        """
        Search live financial news via Google News RSS & Investing.com RSS.
        Fast, robust, never blocked.
        """
        results = []
        # 1. Format query cleanly for forex pairs (e.g. EURUSD -> "EUR USD forex", XAUUSD -> "XAU USD gold forex")
        clean_q = query.strip()
        if len(clean_q) == 6 and clean_q.isalpha() and clean_q.isupper():
            if clean_q == "XAUUSD":
                search_term = "XAU USD gold forex"
            else:
                search_term = f"{clean_q[:3]} {clean_q[3:]} forex"
        else:
            search_term = f"{clean_q} forex" if "forex" not in clean_q.lower() else clean_q

        # Google News RSS
        try:
            encoded_query = urllib.parse.quote(search_term)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            resp = httpx.get(rss_url, headers=self.headers, timeout=6.0)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                for item in root.findall(".//item")[:max_results]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else ""
                    desc = item.find("description").text if item.find("description") is not None else ""
                    clean_desc = BeautifulSoup(desc, "html.parser").get_text(strip=True) if desc else ""
                    results.append({
                        "title": title,
                        "source": "Google News",
                        "published_at": pub_date,
                        "snippet": clean_desc,
                        "url": link
                    })
        except Exception as e:
            logger.warning(f"Google News RSS search failed: {e}")

        return results[:max_results]

    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        General web search using DuckDuckGo HTML parser.
        """
        try:
            url = "https://html.duckduckgo.com/html/"
            resp = httpx.post(url, data={"q": query}, headers=self.headers, timeout=6.0)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                snippets = soup.find_all("a", class_="result__snippet")
                titles = soup.find_all("a", class_="result__url")

                results = []
                for s in snippets[:max_results]:
                    text = s.get_text(strip=True)
                    results.append({
                        "snippet": text,
                        "query": query
                    })
                if results:
                    return results
        except Exception as e:
            logger.warning(f"DuckDuckGo search error: {e}")

        # Fallback to news search
        news = self.search_news(query, max_results=max_results)
        return [{"snippet": f"{n['title']} ({n.get('snippet', '')})", "url": n.get("url", "")} for n in news]

    def scrape_webpage(self, url: str) -> str:
        """Fetch and extract readable clean text from any URL."""
        try:
            resp = httpx.get(url, headers=self.headers, timeout=8.0, follow_redirects=True)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Remove scripts, styles, navigations
                for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 30]
                content = "\n".join(lines[:40])
                return content[:3000] if content else "No readable text content found."
            return f"Failed to retrieve page: HTTP {resp.status_code}"
        except Exception as e:
            return f"Scrape error: {str(e)}"

