# scraping_manager.py (Fixes for ScrapeResponse Import, Firecrawl NameError & Params, Playwright Selectors)
import asyncio
import logging
import json
import re
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime
from processing import ContentProcessor
# Scraping Libraries
from firecrawl import FirecrawlApp
import requests as req_lib
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from urllib.parse import urljoin
from bs4 import BeautifulSoup
# browser-use imports (Keep Agent structure)
try:
    from browser_use import Agent, Browser, BrowserConfig
    BROWSER_USE_AVAILABLE = False
    logger_browser_use_status = logging.getLogger(__name__)
    # Log success only if import worked
    # logger_browser_use_status.info("browser-use library successfully imported.") # Keep log minimal
except ImportError:
    BROWSER_USE_AVAILABLE = False
    logger_browser_use_status = logging.getLogger(__name__)
    # Keep the prominent error message as the runtime import IS failing in the user's execution context
    logger_browser_use_status.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger_browser_use_status.error("! BROWSER-USE library NOT FOUND or import failed AT RUNTIME.")
    logger_browser_use_status.error("! BrowserUseManager WILL BE UNAVAILABLE.")
    logger_browser_use_status.error("! ===> CONFIRMED INSTALLED but NOT FOUND by script execution! <===")
    logger_browser_use_status.error("! CHECK HOW SCRIPT IS RUN (IDE config? Terminal env activation?)")
    logger_browser_use_status.error("! Ensure it uses Python from: .../AskMyProf/.venv/bin/python")
    logger_browser_use_status.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


# Project Configuration
import config

# Setup main logger for this module
logger = logging.getLogger(__name__)

# Initialize Gemini model
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
gemini_model = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    api_key=SecretStr(config.GEMINI_API_KEY)
)

# --- Content Validator (Class definition remains unchanged) ---
class ContentValidator:
    """Validates the adequacy of scraped content."""
    @staticmethod
    def is_content_adequate(clean_content_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if not clean_content_dict:
            logger.warning("Content validation failed: No sections extracted.")
            return False, config.REQUIRED_SECTIONS

        missing_required = []
        for section in config.REQUIRED_SECTIONS:
            if section not in clean_content_dict:
                missing_required.append(section)
            else:
                content = clean_content_dict[section]
                if isinstance(content, str):
                    if not content.strip():
                        missing_required.append(section)
                elif isinstance(content, list):
                    if not content:
                        missing_required.append(section)
                elif content is None:
                    missing_required.append(section)

        if missing_required:
            logger.warning(f"Content validation warning: Missing required sections: {missing_required}. Checking total length...")

        total_length = 0
        for content in clean_content_dict.values():
            if isinstance(content, str):
                total_length += len(content)
            elif isinstance(content, list):
                total_length += sum(len(str(item)) for item in content)
            elif content is not None:
                total_length += len(str(content))

        if total_length < config.MIN_CONTENT_LENGTH:
            logger.error(f"Content validation failed: Total content length ({total_length}) is less than minimum ({config.MIN_CONTENT_LENGTH}).")
            return False, missing_required

        if not missing_required:
            logger.info(f"Content validation passed (Length: {total_length}). All required sections found.")
        else:
            logger.info(f"Content validation passed (Length: {total_length}, but required sections {missing_required} were missing/empty).")
        return True, missing_required

# --- Scraping Managers ---

class FirecrawlManager:
    """Manages scraping using the Firecrawl API."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Firecrawl API key is required.")
        self.api_key = str(api_key) if api_key else None
        if not self.api_key:
             raise ValueError("Firecrawl API key is invalid or missing after check.")
        self.app = FirecrawlApp(api_key=self.api_key)
        logger.info("FirecrawlManager initialized.")

    def scrape(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrapes a single URL using Firecrawl.
        Args:
            url: The URL to scrape.
        Returns:
            The raw scrape result from Firecrawl (usually dict with 'markdown', 'html', or 'data'), or None on failure.
        """
        logger.info(f"Attempting Firecrawl scrape for: {url}")
        try:
            # *** MODIFICATION START ***
            # Correctly structure parameters according to Firecrawl API requirements.
            # 'maxDepth' goes inside 'crawlerOptions'.
            # Removed 'data' from formats as it's invalid for the scrape endpoint.
            params = {
                'pageOptions': {
                    'onlyMainContent': config.FIRECRAWL_CONFIG_REFERENCE['pageOptions'].get('onlyMainContent', True),
                    'formats': ['markdown', 'html'], # Corrected formats list
                    'waitFor': config.FIRECRAWL_CONFIG_REFERENCE['pageOptions'].get('waitFor')
                },
                'crawlerOptions': {
                    'maxDepth': config.FIRECRAWL_CONFIG_REFERENCE['crawlerOptions'].get('maxDepth', 1)
                }
            }

            logger.debug(f"Calling Firecrawl scrape_url with URL: {url} and correctly structured Params: {params}")

            # The firecrawl-py library expects separate kwargs, so we pass them unpacked.
            scrape_result = self.app.scrape_url(
                url=url,
                params=params, # Pass the structured params object
                timeout=config.FIRECRAWL_TIMEOUT
            )
            # *** MODIFICATION END ***

            if scrape_result and ('markdown' in scrape_result or 'html' in scrape_result):
                if 'markdown' in scrape_result and scrape_result['markdown']:
                    logger.info(f"Firecrawl scrape successful (Markdown): {url}")
                    return {'markdown': scrape_result['markdown']}
                elif 'html' in scrape_result and scrape_result['html']:
                    logger.info(f"Firecrawl scrape successful (HTML): {url}")
                    return {'html': scrape_result['html']}
                else:
                    logger.warning(f"Firecrawl scrape for {url} returned empty content for expected formats.")
                    return None
            else:
                logger.warning(f"Firecrawl scrape failed or returned unexpected format for {url}. Result: {scrape_result}")
                return None
                
        except req_lib.exceptions.HTTPError as e:
            error_details = "No specific error details in response."
            status_code = e.response.status_code if e.response is not None else "Unknown"
            if e.response is not None:
                try:
                    error_details = e.response.json()
                except json.JSONDecodeError:
                    error_details = e.response.text[:500]
            logger.error(f"Firecrawl API HTTP error for {url}: Status code {status_code}. {e}. Details: {error_details}", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"Firecrawl unexpected error for {url}: {e}", exc_info=True)
            return None
        
# --- BrowserUseManager (Class definition remains unchanged) ---
class BrowserUseManager:
    """Manages scraping using browser-use via the Agent pattern."""
    def __init__(self):
        if BROWSER_USE_AVAILABLE:
            logger.info("BrowserUseManager initialized (library available).")
        else:
            logger.warning("BrowserUseManager initialized BUT library is unavailable AT RUNTIME. Scrape calls will fail.")

    async def _follow_and_extract_links(self, page_content: str, agent: Agent) -> Dict[str, Any]:
        """Follows relevant links and extracts their content."""
        additional_content = {}
        
        # Keywords to look for in links
        relevant_keywords = [
            'publications', 'research', 'teaching', 'courses', 
            'projects', 'publications', 'papers', 'bio', 'about',
            'education', 'experience', 'service', 'awards'
        ]
        
        # Extract all links from the page
        links = re.findall(r'href="([^"]+)"', page_content)
        
        for link in links:
            # Check if link contains any relevant keywords
            if any(keyword in link.lower() for keyword in relevant_keywords):
                try:
                    logger.info(f"Following link: {link}")
                    await agent.browser.go_to_url(link)
                    await asyncio.sleep(2)  # Wait for page load
                    
                    # Extract content from the linked page
                    link_content = await agent.browser.extract_content({
                        "goal": "Extract the main content from this page",
                        "should_strip_link_urls": True
                    })
                    
                    if link_content:
                        # Use the last part of the URL as a key
                        key = link.split('/')[-1].split('.')[0]
                        additional_content[key] = link_content
                        logger.info(f"Successfully extracted content from {link}")
                except Exception as e:
                    logger.warning(f"Failed to extract content from {link}: {e}")
        
        return additional_content

    async def scrape(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrapes a URL using the browser-use Agent."""
        if not BROWSER_USE_AVAILABLE:
            logger.error("Cannot scrape with BrowserUseManager: browser-use library not available in script's runtime environment.")
            return None

        logger.info(f"Attempting browser-use Agent scrape for: {url}")
        browser_instance = None
        agent = None
        try:
            browser_config_options = {}
            if config.BROWSER_USE_CONFIG.get('use_proxy') and config.BROWSER_USE_CONFIG.get('proxy_url'):
                browser_config_options['proxy'] = config.BROWSER_USE_CONFIG['proxy_url']
                logger.info("Configuring browser-use with proxy.")

            extraction_format = config.BROWSER_USE_CONFIG.get('extraction_strategy', 'markdown')
            task_prompt = f"""
            Objective: Scrape the main academic profile content from the URL: {url}.
            Instructions:
            1. Navigate to {url}.
            2. Handle cookie banners if they appear using selectors like {config.BROWSER_USE_CONFIG['interaction_patterns']['cookie_consent']}.
            3. Extract the primary content (bio, research, publications, teaching, etc.) using hints like {config.BROWSER_USE_CONFIG['element_prioritization']}.
            4. Return the extracted content in {extraction_format} format. If markdown isn't feasible, return full page HTML.
            Output: Provide result as JSON object with key '{extraction_format}' or 'html'. Example: `{{"{extraction_format}": "..."}}` or `{{"html": "..."}}`.
            """

            browser_instance = Browser(config=BrowserConfig(**browser_config_options))
            agent = Agent(
                browser=browser_instance,
                task=task_prompt,
                llm=gemini_model
            )

            logger.info(f"Running browser-use Agent for {url}...")
            history = await agent.run()
            result_data = history.final_result() if history else None
            logger.info(f"browser-use Agent finished for {url}.")

            if isinstance(result_data, dict):
                content_key = extraction_format if extraction_format in result_data and result_data[extraction_format] else None
                if not content_key and 'html' in result_data and result_data['html']: 
                    content_key = 'html'

                if content_key:
                    # Get the page content
                    page_content = result_data[content_key]
                    
                    # Follow and extract content from relevant links
                    additional_content = await self._follow_and_extract_links(page_content, agent)
                    
                    # Merge additional content with main content
                    if additional_content:
                        result_data['additional_content'] = additional_content
                        logger.info(f"Added content from {len(additional_content)} additional pages")
                    
                    content_length = len(result_data[content_key])
                    logger.info(f"browser-use scrape successful ({content_key}, length: {content_length}): {url}")
                    return result_data
                else:
                    logger.warning(f"browser-use Agent returned dict but missing expected keys/content for {url}. Keys: {list(result_data.keys())}")
                    return None
            elif isinstance(result_data, str) and result_data.strip():
                logger.info(f"browser-use Agent returned raw string, assuming {extraction_format}: {url}")
                return {extraction_format: result_data}
            else:
                logger.warning(f"browser-use Agent scrape failed or returned unexpected/empty data for {url}. Type: {type(result_data)}")
                return None

        except ImportError:
            logger.error("BrowserUseManager cannot run: browser-use components not found at runtime (should have been caught earlier).")
            return None
        except AttributeError as e:
            logger.error(f"browser-use AttributeError for {url}: {e}. Possible API mismatch. Name='{getattr(e, 'name', 'N/A')}', Obj='{getattr(e, 'obj', 'N/A')}'", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"browser-use unexpected error for {url}: {e}", exc_info=True)
            return None
        finally:
            if browser_instance and hasattr(browser_instance, 'close'):
                try:
                    logger.debug("Final attempt to close browser-use instance in finally block.")
                    await browser_instance.close()
                except Exception as close_err:
                    logger.error(f"Error closing browser-use instance in finally block: {close_err}")

# --- PlaywrightManager (Class definition remains unchanged from last version) ---
# It was working correctly in the last log.
class PlaywrightManager:
    """Manages scraping using Playwright for fine-grained control."""
    async def scrape(self, url: str, specific_actions: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Scrapes a URL using Playwright, with added logic to discover and follow
        relevant academic links (e.g., to Publications, Research) to gather more complete data.
        """
        logger.info(f"Attempting enhanced Playwright scrape for: {url}")
        playwright = None
        browser = None
        context = None
        page = None
        resources_closed = False

        try:
            playwright = await async_playwright().start()
            launch_options = {
                'headless': config.PLAYWRIGHT_HEADLESS,
            }
            if config.PLAYWRIGHT_USE_PROXY and config.PLAYWRIGHT_PROXY_CONFIG:
                launch_options['proxy'] = config.PLAYWRIGHT_PROXY_CONFIG
                logger.info("Launching Playwright with proxy.")

            browser = await playwright.chromium.launch(**launch_options)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                java_script_enabled=True,
                ignore_https_errors=True,
                viewport={'width': 1920, 'height': 1080},
            )
            page = await context.new_page()
            page.set_default_navigation_timeout(config.PLAYWRIGHT_TIMEOUT)
            page.set_default_timeout(config.PLAYWRIGHT_TIMEOUT)

            # --- Stage 1: Scrape the main page ---
            logger.debug(f"Navigating to main page: {url}")
            await page.goto(url, wait_until='domcontentloaded')
            logger.debug(f"Navigation to {url} completed.")
            
            # (Your existing cookie-clicking and scrolling logic can remain here)
            # ...

            initial_html = await page.content()
            combined_html = [initial_html]
            logger.info(f"Successfully scraped initial page content (length: {len(initial_html)}).")

            # --- Stage 2: Discover and scrape relevant sub-pages ---
            soup = BeautifulSoup(initial_html, 'html.parser')
            links = soup.find_all('a', href=True)
            
            relevant_keywords = [
                'publication', 'paper', 'research', 'project', 'cv', 'bio', 
                'teaching', 'course', 'service', 'award', 'group', 'lab'
            ]
            
            visited_urls = {url.rstrip('/')}
            
            logger.info(f"Found {len(links)} links. Searching for relevant pages...")
            
            for link in links:
                link_text = link.get_text(strip=True).lower()
                link_href = link['href'].lower()
                
                # Check if the link seems relevant based on keywords
                if any(keyword in link_text for keyword in relevant_keywords) or any(keyword in link_href for keyword in relevant_keywords):
                    absolute_url = urljoin(url, link['href']).rstrip('/')
                    
                    if absolute_url.startswith('http') and absolute_url not in visited_urls:
                        visited_urls.add(absolute_url)
                        logger.info(f"Found relevant link: '{link_text}' -> Navigating to {absolute_url}")
                        
                        try:
                            await page.goto(absolute_url, wait_until='domcontentloaded')
                            await page.wait_for_timeout(2000) # Wait for page to settle
                            
                            sub_page_html = await page.content()
                            combined_html.append(f"\n\n\n\n" + sub_page_html)
                            logger.info(f"Successfully scraped sub-page. Total content parts now: {len(combined_html)}")
                            
                        except PlaywrightTimeoutError:
                            logger.warning(f"Timeout when trying to navigate to sub-page: {absolute_url}")
                        except Exception as e:
                            logger.error(f"Error scraping sub-page {absolute_url}: {e}")

            # --- Stage 3: Finalize and Return ---
            final_html_content = "\n".join(combined_html)
            logger.debug("Closing Playwright resources...")
            if page and not page.is_closed(): await page.close()
            if context: await context.close()
            if browser and browser.is_connected(): await browser.close()
            if playwright: await playwright.stop()
            resources_closed = True
            logger.debug("Playwright resources closed.")

            if final_html_content:
                logger.info(f"Playwright enhanced scrape successful (Total HTML length: {len(final_html_content)}): {url}")
                return {'html': final_html_content}
            else:
                logger.warning(f"Playwright scrape returned empty or minimal HTML for {url}.")
                return None
        except Exception as e:
            logger.error(f"Playwright unexpected error for {url}: {e}", exc_info=True)
            return None
        finally:
            if not resources_closed:
                logger.warning("Closing Playwright resources in finally block due to earlier exit/error.")
                # Add try-except around each close in finally for resilience
                if page and not page.is_closed():
                    try: await page.close()
                    except Exception as e: logger.error(f"Error closing page in finally: {e}")
                if context:
                    try: await context.close()
                    except Exception as e: logger.error(f"Error closing context in finally: {e}")
                if browser and browser.is_connected():
                    try: await browser.close()
                    except Exception as e: logger.error(f"Error closing browser in finally: {e}")
                if playwright:
                    try: await playwright.stop()
                    except Exception as e: logger.error(f"Error stopping playwright in finally: {e}")


# --- Orchestrator (Class definition remains unchanged) ---
class ScrapeOrchestrator:
    """Orchestrates the scraping and new processing pipeline."""
    def __init__(self):
        self.firecrawl_manager = None
        try:
            if config.FIRE_CRAWL_API_KEY:
                self.firecrawl_manager = FirecrawlManager(api_key=config.FIRE_CRAWL_API_KEY)
                logger.info("Orchestrator: FirecrawlManager initialized.")
            else:
                logger.warning("Orchestrator: Firecrawl API key missing. FirecrawlManager disabled.")
        except ValueError as e:
            logger.error(f"Orchestrator: Failed to initialize FirecrawlManager: {e}")
            self.firecrawl_manager = None

        self.browser_use_manager = BrowserUseManager()
        self.playwright_manager = PlaywrightManager()
        self.processor = ContentProcessor() # Initialize the new processor
        logger.info("ScrapeOrchestrator initialized with integrated ContentProcessor.")

    async def get_and_process_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Main entry point. Scrapes a URL, then processes the content to return a final, structured JSON.
        """
        raw_output = None
        source_tool = None

        # --- Stage 1: Scraping (same logic as before) ---
        # 1. Try Firecrawl
        if self.firecrawl_manager:
            logger.info(f"Orchestrator: ---> [1] Trying Firecrawl for {url}")
            raw_output = self.firecrawl_manager.scrape(url)
            if raw_output and (raw_output.get('markdown') or raw_output.get('html') or raw_output.get('data')):
                source_tool = 'firecrawl'
                logger.info(f"Orchestrator: Firecrawl succeeded for {url}.")
            else:
                raw_output = None # Ensure it's None if content is empty

        # 2. Try Browser-Use (if no success yet and available)
        if not raw_output and BROWSER_USE_AVAILABLE:
            logger.info(f"Orchestrator: ---> [2] Trying browser-use for {url}")
            raw_output = await self.browser_use_manager.scrape(url)
            if raw_output:
                source_tool = 'browser-use'
                logger.info(f"Orchestrator: browser-use succeeded for {url}.")

        # 3. Try Playwright (if still no success)
        if not raw_output:
            logger.info(f"Orchestrator: ---> [3] Trying Playwright for {url}")
            raw_output = await self.playwright_manager.scrape(url)
            if raw_output:
                source_tool = 'playwright'
                logger.info(f"Orchestrator: Playwright succeeded for {url}.")

        # --- Stage 2: Processing ---
        if raw_output and source_tool:
            logger.info(f"Content scraped via {source_tool}. Handing off to ContentProcessor.")
            
            final_output = self.processor.process_and_structure(raw_output, source_url=url)
            
            if final_output:
                # Enrich metadata
                final_output["metadata"]["timestamp"] = datetime.now().isoformat()
                final_output["metadata"]["scraping_tool_used"] = source_tool
                final_output["metadata"]["status"] = "success"
                logger.info(f"Successfully processed and structured data for {url}.")
                return final_output
            else:
                logger.error(f"Content processing failed for scraped data from {url}.")
                return None
        else:
            logger.error(f"All scraping methods failed for {url}. No content to process.")
            return None