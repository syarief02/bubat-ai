import httpx
import asyncio
try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    AsyncWebCrawler = None

try:
    import html2text
except ImportError:
    html2text = None

from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any
from loguru import logger
import json
from pathlib import Path
import re
import urllib.parse
import xml.etree.ElementTree as ET

try:
    from core.web_surfer import WebSurfer
except ImportError:
    try:
        from forex_local_agent.core.web_surfer import WebSurfer
    except ImportError:
        WebSurfer = None

try:
    from learning.skills.economic_calendar_filter import get_economic_events_summary, calendar_filter
except ImportError:
    try:
        from forex_local_agent.learning.skills.economic_calendar_filter import get_economic_events_summary, calendar_filter
    except ImportError:
        get_economic_events_summary = None
        calendar_filter = None


class SentimentEngine:
    def __init__(self, config_path: str | Path):
        """
        Initialize the Sentiment Engine.
        
        Args:
            config_path: Path to the JSON configuration file.
        """
        self.config_path = Path(config_path)
        self.searxng_url = "http://localhost:8080"
        self.max_news_tokens = 4000
        self.searxng_available = None  # None: untried, True: online, False: offline
        self.web_surfer = WebSurfer() if WebSurfer else None
        
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from the specified path."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.searxng_url = config.get("searxng_url", self.searxng_url)
                    self.max_news_tokens = config.get("trading", {}).get("max_news_tokens", self.max_news_tokens)
            except Exception as e:
                logger.error(f"Failed to load config from {self.config_path}: {e}")
        else:
            logger.warning(f"Config file not found at {self.config_path}. Using default settings.")

    async def _probe_searxng(self) -> bool:
        """Check once if SearXNG is reachable without retry delays."""
        if not self.searxng_url:
            self.searxng_available = False
            return False
            
        try:
            async with httpx.AsyncClient(timeout=0.6) as client:
                resp = await client.get(f"{self.searxng_url}/", follow_redirects=True)
                if resp.status_code < 500:
                    self.searxng_available = True
                    return True
        except Exception:
            pass
            
        self.searxng_available = False
        logger.info("SearXNG offline (Docker not running). Fast-routing to live WebSurfer news feed.")
        return False

    async def _fetch_with_retry(self, url: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 2) -> httpx.Response:
        """Fetch URL with minimal retries, immediately aborting on connection failure."""
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    return response
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                logger.debug(f"Host unreachable for URL {url}: {e}")
                raise
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.debug(f"Request failed for {url}: {e}")
                    raise
                await asyncio.sleep(0.5)
        raise RuntimeError(f"Failed to fetch {url}")

    async def search_news(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for recent forex news articles.
        Uses WebSurfer (Google News RSS & Investing.com) directly when SearXNG is offline.
        
        Args:
            query: The search query string.
            max_results: Maximum number of results to return.
            
        Returns:
            List of dictionaries containing news article details.
        """
        logger.info(f"Searching news for query: {query}")
        
        # Probe SearXNG once if not probed yet
        if self.searxng_available is None:
            await self._probe_searxng()
            
        if self.searxng_available:
            try:
                params = {
                    "q": f"{query} forex news",
                    "format": "json"
                }
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"{self.searxng_url}/search", params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", [])
                        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=18)
                        filtered_results = []
                        for item in results:
                            pub_date_str = item.get("publishedDate")
                            pub_date = None
                            if pub_date_str:
                                try:
                                    pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                                    if pub_date.tzinfo is None:
                                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                                except ValueError:
                                    pass
                            if pub_date and pub_date < cutoff_time:
                                continue
                            filtered_results.append({
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "published_date": pub_date_str or datetime.now(timezone.utc).isoformat(),
                                "snippet": item.get("content", "")
                            })
                            if len(filtered_results) >= max_results:
                                break
                        if filtered_results:
                            return filtered_results
            except Exception as e:
                logger.debug(f"SearXNG query error: {e}. Falling back to live WebSurfer.")
                self.searxng_available = False

        # Live WebSurfer fallback (Google News RSS + Investing.com RSS)
        return await self._fallback_search_news(query, max_results)

    async def _fallback_search_news(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Fast live financial news retrieval via WebSurfer / Google News RSS."""
        try:
            if self.web_surfer:
                news = await asyncio.to_thread(self.web_surfer.search_news, query=query, max_results=max_results)
                if news:
                    logger.info(f"Retrieved {len(news)} live news headlines via WebSurfer.")
                    return [
                        {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "published_date": item.get("published_at", datetime.now(timezone.utc).isoformat()),
                            "snippet": item.get("snippet", "")
                        }
                        for item in news
                    ]
        except Exception as e:
            logger.debug(f"WebSurfer search failed: {e}")

        # Direct Google News RSS fallback
        try:
            encoded_query = urllib.parse.quote(f"{query} forex")
            feed_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            async with httpx.AsyncClient(timeout=5.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
                resp = await client.get(feed_url)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.text)
                    items = root.findall(".//item")
                    results = []
                    for it in items[:max_results]:
                        title = it.find("title").text if it.find("title") is not None else ""
                        link = it.find("link").text if it.find("link") is not None else ""
                        pub_date = it.find("pubDate").text if it.find("pubDate") is not None else ""
                        desc = it.find("description").text if it.find("description") is not None else ""
                        clean_desc = BeautifulSoup(desc, "html.parser").get_text(strip=True) if desc else ""
                        results.append({
                            "title": title,
                            "url": link,
                            "published_date": pub_date or datetime.now(timezone.utc).isoformat(),
                            "snippet": clean_desc
                        })
                    logger.info(f"Retrieved {len(results)} live news headlines via Google News RSS.")
                    return results
        except Exception as err:
            logger.debug(f"Fallback news retrieval failed for {query} ({type(err).__name__}: {err})")
        return []

    async def scrape_article(self, url: str) -> str:
        """
        Scrape article content with automatic fallback.
        
        Args:
            url: URL of the article to scrape.
            
        Returns:
            Clean text or markdown of the article.
        """
        if AsyncWebCrawler is not None:
            try:
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await crawler.arun(url=url)
                    if result and result.markdown:
                        return result.markdown
            except Exception as e:
                logger.debug(f"Crawl4AI failed for {url}: {e}")

        # Fallback to direct HTTP fetch + markdown/text extraction
        try:
            async with httpx.AsyncClient(timeout=4.0, follow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                        tag.decompose()
                    
                    if html2text:
                        h = html2text.HTML2Text()
                        h.ignore_links = True
                        h.ignore_images = True
                        return h.handle(str(soup))
                    else:
                        return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.debug(f"Fallback HTTP scraping failed for {url}: {e}")

        return ""

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit cleanly at sentence boundaries."""
        max_words = int(max_tokens * 0.75)
        words = text.split()
        
        if len(words) <= max_words:
            return text
            
        truncated_words = words[:max_words]
        truncated_text = " ".join(truncated_words)
        
        match = re.search(r'([.?!])\s+[A-Z0-9]', truncated_text[::-1])
        if match:
            cutoff = len(truncated_text) - match.start()
            return truncated_text[:cutoff].strip()
            
        return truncated_text + "..."

    async def get_live_news(self, symbol: str) -> Dict[str, Any]:
        """
        Get and aggregate live news for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD').
            
        Returns:
            Dictionary with aggregated news and content.
        """
        logger.info(f"Fetching live news for symbol: {symbol}")
        news_items = await self.search_news(symbol, max_results=3)
        
        headlines = [it["title"] for it in news_items]
        
        # Concurrently build article snippets without hanging on redirects
        async def _fetch_item_text(item: Dict[str, Any]) -> str:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            url = item.get("url", "")
            content = ""
            # Only scrape if URL is a direct web article, not a Google redirect
            if url and not url.startswith("https://news.google.com/"):
                try:
                    content = await asyncio.wait_for(self.scrape_article(url), timeout=2.5)
                except Exception:
                    content = ""
            if content and len(content.strip()) > 80:
                return f"## {title}\nSummary: {snippet}\n\n{content}"
            return f"## {title}\nSummary: {snippet}"

        parts = await asyncio.gather(*[_fetch_item_text(it) for it in news_items])
        combined_text = "\n\n".join(parts)
        truncated_text = self._truncate_to_tokens(combined_text, self.max_news_tokens)

        # Inject real-time macroeconomic calendar events summary if available
        cal_summary = ""
        if get_economic_events_summary:
            try:
                cal_summary = get_economic_events_summary(symbol)
                if cal_summary:
                    truncated_text = f"=== MACROECONOMIC CALENDAR FOR {symbol} ===\n{cal_summary}\n\n=== LIVE NEWS ARTICLES ===\n{truncated_text}"
            except Exception as e:
                logger.debug(f"Could not fetch calendar summary for {symbol}: {e}")
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "news_count": len(news_items),
            "headlines": headlines,
            "full_text": truncated_text,
            "economic_calendar_summary": cal_summary
        }

    async def get_economic_calendar(self) -> List[Dict[str, Any]]:
        """
        Get today's economic calendar events.
        Prioritizes the live ForexFactory calendar feed.
        
        Returns:
            List of dictionaries containing event details.
        """
        logger.info("Fetching today's economic calendar...")

        if calendar_filter:
            try:
                raw_events = calendar_filter.refresh_calendar()
                if raw_events:
                    formatted_events = []
                    for ev in raw_events[:20]:
                        formatted_events.append({
                            "time": ev.get("time_utc", datetime.now(timezone.utc).isoformat()),
                            "event": ev.get("title", "Economic Event"),
                            "currency": ev.get("currency", "USD"),
                            "impact": ev.get("impact", "Medium"),
                            "forecast": ev.get("forecast", ""),
                            "previous": ev.get("previous", ""),
                            "source_url": "ForexFactory Live Feed"
                        })
                    logger.info(f"Loaded {len(formatted_events)} events from ForexFactory calendar feed.")
                    return formatted_events
            except Exception as e:
                logger.warning(f"ForexFactory calendar parse failed: {e}. Falling back...")
        
        if self.searxng_available is None:
            await self._probe_searxng()
            
        if self.searxng_available:
            try:
                query = "today's economic calendar forex events"
                params = {
                    "q": query,
                    "format": "json"
                }
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"{self.searxng_url}/search", params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", [])
                        events = []
                        for item in results[:5]:
                            events.append({
                                "time": datetime.now(timezone.utc).isoformat(),
                                "event": item.get("title", "Unknown Event"),
                                "currency": "USD",
                                "impact": "Medium",
                                "forecast": "",
                                "previous": "",
                                "source_url": item.get("url", "")
                            })
                        return events
            except Exception as e:
                logger.debug(f"SearXNG calendar error: {e}")
                self.searxng_available = False
                
        return await self._fallback_economic_calendar()

    async def _fallback_economic_calendar(self) -> List[Dict[str, Any]]:
        """Fallback calendar events via live financial news / calendar feed."""
        try:
            if self.web_surfer:
                news = await asyncio.to_thread(self.web_surfer.search_news, query="economic calendar forex high impact", max_results=5)
                events = []
                for item in news:
                    events.append({
                        "time": item.get("published_at", datetime.now(timezone.utc).isoformat()),
                        "event": item.get("title", "Economic Event"),
                        "currency": "USD",
                        "impact": "High" if any(w in item.get("title", "").upper() for w in ["CPI", "FED", "NFP", "RATE", "INFLATION", "GDP", "ECB", "BOE", "BOJ"]) else "Medium",
                        "forecast": "",
                        "previous": "",
                        "source_url": item.get("url", "")
                    })
                return events
        except Exception as e:
            logger.debug(f"Fallback economic calendar failed: {e}")
        return []
