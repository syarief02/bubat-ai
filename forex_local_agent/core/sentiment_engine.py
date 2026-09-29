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
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from loguru import logger
import json
from pathlib import Path
import re

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

    async def _fetch_with_retry(self, url: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> httpx.Response:
        """Fetch URL with retries and exponential backoff."""
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    return response
            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1} failed for URL {url}: {e}. Retrying in {wait_time}s...")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed for URL {url}")
                    raise
                await asyncio.sleep(wait_time)
        raise RuntimeError(f"Failed to fetch {url}")

    async def search_news(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for recent forex news articles using SearXNG.
        
        Args:
            query: The search query string.
            max_results: Maximum number of results to return.
            
        Returns:
            List of dictionaries containing news article details.
        """
        logger.info(f"Searching news for query: {query}")
        try:
            params = {
                "q": f"{query} forex news",
                "format": "json"
            }
            response = await self._fetch_with_retry(f"{self.searxng_url}/search", params=params)
            data = response.json()
            results = data.get("results", [])
            
            cutoff_time = datetime.utcnow() - timedelta(hours=12)
            filtered_results = []
            
            for item in results:
                pub_date_str = item.get("publishedDate")
                pub_date = None
                if pub_date_str:
                    try:
                        # Attempt to parse common format or fallback
                        pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        pub_date = pub_date.replace(tzinfo=None)
                    except ValueError:
                        pass
                
                # If date is parsed and older than 12h, skip
                if pub_date and pub_date < cutoff_time:
                    continue
                    
                filtered_results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "published_date": pub_date_str or datetime.utcnow().isoformat(),
                    "snippet": item.get("content", "")
                })
                
                if len(filtered_results) >= max_results:
                    break
                    
            return filtered_results
        except Exception as e:
            logger.error(f"Error searching news for {query}: {e}")
            return []

    async def scrape_article(self, url: str) -> str:
        """
        Scrape article content using Crawl4AI with automatic HTTP fallback.
        
        Args:
            url: URL of the article to scrape.
            
        Returns:
            Markdown formatted text of the article.
        """
        logger.info(f"Scraping article: {url}")
        
        if AsyncWebCrawler is not None:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    async with AsyncWebCrawler(verbose=False) as crawler:
                        result = await crawler.arun(url=url)
                        if result and result.markdown:
                            return result.markdown
                except Exception as e:
                    logger.warning(f"Crawl4AI attempt {attempt + 1} failed for {url}: {e}")
                    await asyncio.sleep(1)

        # Fallback to direct HTTP fetch + markdown extraction
        try:
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Remove scripts, styles, nav, footer
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
            logger.warning(f"Fallback HTTP scraping failed for {url}: {e}")

        return ""

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to approximate token limit cleanly at sentence boundaries.
        
        Args:
            text: Input text to truncate.
            max_tokens: Maximum allowed tokens.
            
        Returns:
            Truncated text.
        """
        max_words = int(max_tokens * 0.75)
        words = text.split()
        
        if len(words) <= max_words:
            return text
            
        truncated_words = words[:max_words]
        truncated_text = " ".join(truncated_words)
        
        # Find the last sentence boundary
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
        
        combined_text = ""
        headlines = []
        
        for item in news_items:
            headlines.append(item["title"])
            content = await self.scrape_article(item["url"])
            combined_text += f"\n\n## {item['title']}\n{content}"
            
        truncated_text = self._truncate_to_tokens(combined_text, self.max_news_tokens)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "news_count": len(news_items),
            "headlines": headlines,
            "full_text": truncated_text
        }

    async def get_economic_calendar(self) -> List[Dict[str, Any]]:
        """
        Get today's economic calendar events.
        
        Returns:
            List of dictionaries containing event details.
        """
        logger.info("Fetching today's economic calendar...")
        try:
            query = "today's economic calendar forex events"
            params = {
                "q": query,
                "format": "json"
            }
            response = await self._fetch_with_retry(f"{self.searxng_url}/search", params=params)
            data = response.json()
            
            results = data.get("results", [])
            events = []
            
            for item in results[:5]: # Take top 5
                events.append({
                    "time": datetime.utcnow().isoformat(), # mock time
                    "event": item.get("title", "Unknown Event"),
                    "currency": "USD", # mock currency
                    "impact": "Medium", # mock impact
                    "forecast": "",
                    "previous": "",
                    "source_url": item.get("url", "")
                })
                
            return events
            
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return []
