# scraping_manager.py (Fixes for ScrapeResponse Import, Firecrawl NameError & Params, Playwright Selectors)
import asyncio
import logging
import json
import re
from typing import Tuple, Optional, Dict, Any, List

# Scraping Libraries
from firecrawl import FirecrawlApp
import requests as req_lib
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

# browser-use imports (Keep Agent structure)
try:
    from browser_use import Agent, Browser, BrowserConfig
    BROWSER_USE_AVAILABLE = True
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
    def is_content_adequate(clean_content_dict: Dict[str, str]) -> Tuple[bool, List[str]]:
        if not clean_content_dict:
            logger.warning("Content validation failed: No sections extracted.")
            return False, config.REQUIRED_SECTIONS

        missing_required = [
            section for section in config.REQUIRED_SECTIONS
            if section not in clean_content_dict or not clean_content_dict[section].strip()
        ]
        if missing_required:
            logger.warning(f"Content validation warning: Missing required sections: {missing_required}. Checking total length...")

        total_length = sum(len(content) for content in clean_content_dict.values())
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
        # Ensure API key is string, handle potential None from getenv more robustly
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
            # Prepare parameters based on config
            params_to_pass = {
                'onlyMainContent': config.FIRECRAWL_CONFIG_REFERENCE['pageOptions']['onlyMainContent'],
                'formats': config.FIRECRAWL_CONFIG_REFERENCE['pageOptions']['formats'],
                'waitFor': config.FIRECRAWL_CONFIG_REFERENCE['pageOptions']['waitFor'],
                'maxDepth': config.FIRECRAWL_CONFIG_REFERENCE['crawlerOptions']['maxDepth']
            }

            logger.debug(f"Calling Firecrawl scrape_url with URL: {url} and Params: {params_to_pass}")

            scrape_result = self.app.scrape_url(
                url=url,
                timeout=config.FIRECRAWL_TIMEOUT,
                **params_to_pass
            )

            # Original validation logic remains the same
            if scrape_result and ('markdown' in scrape_result or 'html' in scrape_result or 'data' in scrape_result):
                # Prioritize returning 'data' if available and requested/present
                if 'data' in scrape_result and scrape_result['data']:
                    logger.info(f"Firecrawl scrape successful (structured data found): {url}")
                    return {'data': scrape_result['data']}
                # Then markdown if available and requested/present
                elif 'markdown' in scrape_result and scrape_result['markdown']:
                    logger.info(f"Firecrawl scrape successful (Markdown): {url}")
                    return {'markdown': scrape_result['markdown']}
                # Then HTML if available and requested/present
                elif 'html' in scrape_result and scrape_result['html']:
                    logger.info(f"Firecrawl scrape successful (HTML): {url}")
                    return {'html': scrape_result['html']}
                else:
                    # This case handles if the result dict exists but the expected keys have empty content
                    logger.warning(f"Firecrawl scrape for {url} returned empty content for expected formats. Result keys: {list(scrape_result.keys())}")
                    return None
            else:
                logger.warning(f"Firecrawl scrape failed or returned unexpected format for {url}. Result: {scrape_result}")
                return None
        # Catch the specific HTTPError from requests library used by firecrawl-py
        except req_lib.exceptions.HTTPError as e:
            # Log the specific Firecrawl error message if available in the response
            error_details = "No specific error details in response."
            status_code = e.response.status_code if e.response is not None else "Unknown"
            if e.response is not None:
                try:
                    error_details = e.response.json()
                except json.JSONDecodeError:
                    error_details = e.response.text[:500] # Log raw text if not JSON
            logger.error(f"Firecrawl API HTTP error for {url}: Status code {status_code}. {e}. Details: {error_details}", exc_info=False) # exc_info=False to avoid redundant traceback
            return None
        except Exception as e:
            # Catch other potential errors during the scrape call
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
        """ Scrapes a URL using Playwright. """
        logger.info(f"Attempting Playwright scrape for: {url}")
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

            logger.debug(f"Navigating to {url} with timeout {config.PLAYWRIGHT_TIMEOUT}ms")
            await page.goto(url, wait_until='domcontentloaded')
            logger.debug(f"Navigation to {url} completed.")

            # --- Click cookie banners ---
            logger.info("Playwright attempting to click cookie consent buttons...")
            cookie_selectors = config.BROWSER_USE_CONFIG['interaction_patterns'].get('cookie_consent', [])
            clicked_cookie_banner = False
            cookie_find_timeout = 7000
            action_timeout = config.PLAYWRIGHT_TIMEOUT

            for selector in cookie_selectors:
                 if clicked_cookie_banner: break
                 if ":contains(" in selector:
                      logger.warning(f"Skipping potentially invalid Playwright selector from config: {selector}")
                      continue
                 try:
                     logger.debug(f"Trying cookie selector: {selector}")
                     button = page.locator(selector).first
                     if await button.is_visible(timeout=cookie_find_timeout):
                         logger.debug(f"Found potential cookie button: {selector}")
                         await button.click(timeout=action_timeout)
                         logger.info(f"Playwright clicked cookie consent using selector: {selector}")
                         await page.wait_for_timeout(500)
                         clicked_cookie_banner = True
                         break
                     else:
                         logger.debug(f"Cookie button not visible within timeout: {selector}")
                 except PlaywrightTimeoutError:
                     logger.debug(f"Timeout waiting for cookie element visibility or click action: {selector}")
                 except PlaywrightError as pe:
                      if "SyntaxError" in str(pe) or "valid selector" in str(pe):
                          logger.error(f"Playwright invalid selector syntax provided: '{selector}' - Error: {pe}", exc_info=False)
                      else:
                          logger.warning(f"Playwright error interacting with cookie selector '{selector}': {pe}")
                 except Exception as click_err:
                     logger.warning(f"Playwright unexpected error clicking cookie consent '{selector}': {click_err}")

            if not clicked_cookie_banner:
                 logger.info("Playwright: No cookie banners found or clicked matching configured selectors.")

            # --- Scroll down page ---
            logger.info("Playwright scrolling down page...")
            scroll_attempts = 3
            scroll_pause = 1500
            page_height = await page.evaluate("document.body.scrollHeight")
            logger.debug(f"Initial page height: {page_height}")
            for i in range(scroll_attempts):
                 await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                 await page.wait_for_timeout(scroll_pause)
                 new_height = await page.evaluate("document.body.scrollHeight")
                 logger.debug(f"Scroll attempt {i+1}, new height: {new_height}")
                 if new_height == page_height:
                      logger.info(f"Scrolling stopped after attempt {i+1}, height unchanged.")
                      break
                 page_height = new_height
            logger.info("Playwright scrolling finished.")

            html_content = await page.content()

            # --- Close Resources ---
            logger.debug("Closing Playwright resources...")
            if page and not page.is_closed(): await page.close()
            if context: await context.close()
            if browser and browser.is_connected(): await browser.close()
            if playwright: await playwright.stop()
            resources_closed = True
            logger.debug("Playwright resources closed.")

            if html_content and len(html_content) > 50:
                logger.info(f"Playwright scrape successful (HTML length: {len(html_content)}): {url}")
                return {'html': html_content}
            else:
                logger.warning(f"Playwright scrape returned empty or minimal HTML for {url}.")
                return None
        # (Exception handling remains unchanged)
        except PlaywrightTimeoutError as pte:
             logger.error(f"Playwright timed out for {url}: {pte}", exc_info=False)
             return None
        except TypeError as te:
             logger.error(f"Playwright TypeError (likely incorrect arguments): {te}", exc_info=True)
             return None
        except PlaywrightError as pe:
             if "SyntaxError" in str(pe) or "valid selector" in str(pe):
                 logger.error(f"Playwright encountered invalid selector syntax: {pe}", exc_info=False)
             else:
                 logger.error(f"Playwright execution error for {url}: {pe}", exc_info=True)
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
    """Orchestrates the scraping process using multiple strategies."""
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
        self.validator = ContentValidator()
        logger.info("ScrapeOrchestrator initialized.")

    async def get_content(self, url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Tries scraping methods in order: Firecrawl -> Browser-Use -> Playwright."""
        raw_output = None
        source_tool = None

        # 1. Try Firecrawl
        if self.firecrawl_manager:
            logger.info(f"Orchestrator: ---> [1] Trying Firecrawl for {url}")
            raw_output = self.firecrawl_manager.scrape(url) # Sync call
            if raw_output:
                content_key = next((k for k in ['markdown', 'data', 'html'] if k in raw_output and raw_output[k]), None)
                if content_key:
                     logger.info(f"Orchestrator: Firecrawl succeeded ({content_key} found) for {url}.")
                     source_tool = 'firecrawl'
                     return raw_output, source_tool
                else:
                     logger.warning(f"Orchestrator: Firecrawl returned a response but no content. Falling back.")
                     raw_output = None
            else:
                 logger.warning(f"Orchestrator: Firecrawl scrape method returned None. Falling back.")
        else:
             logger.info("Orchestrator: ---> [1] Skipping Firecrawl (unavailable or failed init).")

        # 2. Try Browser-Use (if available AT RUNTIME)
        if BROWSER_USE_AVAILABLE:
            logger.info(f"Orchestrator: ---> [2] Trying browser-use for {url}")
            raw_output = await self.browser_use_manager.scrape(url) # Async call
            if raw_output:
                source_tool = 'browser-use'
                extraction_format = config.BROWSER_USE_CONFIG.get('extraction_strategy', 'markdown')
                content_key = extraction_format if extraction_format in raw_output and raw_output[extraction_format] else None
                if not content_key and 'html' in raw_output and raw_output['html']: content_key = 'html'

                if content_key:
                    logger.info(f"Orchestrator: browser-use succeeded ({content_key} found) for {url}.")
                    return raw_output, source_tool
                else:
                     logger.warning(f"Orchestrator: browser-use returned empty/unexpected content. Falling back.")
                     raw_output = None; source_tool = None
            else:
                 logger.warning(f"Orchestrator: browser-use scrape method returned None. Falling back.")
        else:
            # Logged at init and import time, keep this minimal
            logger.info("Orchestrator: ---> [2] Skipping browser-use (unavailable in runtime environment).")

        # 3. Try Playwright (Fallback)
        logger.info(f"Orchestrator: ---> [3] Trying Playwright for {url}")
        raw_output = await self.playwright_manager.scrape(url) # Async
        if raw_output:
            source_tool = 'playwright'
            if raw_output.get('html'):
                 logger.info(f"Orchestrator: Playwright succeeded (HTML found) for {url}.")
                 return raw_output, source_tool
            else:
                 logger.warning("Orchestrator: Playwright returned empty HTML content. No more fallbacks.")
                 raw_output = None; source_tool = None
        else:
             logger.warning(f"Orchestrator: Playwright scrape method returned None. No more fallbacks.")


        # 4. If all methods failed
        logger.error(f"Orchestrator: All scraping methods failed to retrieve content for {url}.")
        return None, None