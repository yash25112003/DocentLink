# scraping_manager.py
import asyncio
import logging
from typing import Tuple, Optional, Dict, Any

# Scraping Libraries
from firecrawl import FirecrawlApp
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
# Assuming browser-use has an interface like this. Adjust if the actual API differs.
# If browser-use primarily uses an Agent model, integrating it here might require
# defining a generic task prompt or using its lower-level browser control.
# For simplicity, we'll simulate a direct scrape function.
try:
    # Try importing the main browser-use components
    from browser_use import Browser, BrowserConfig # Adjust imports as needed
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    print("Warning: browser-use library not found or import failed. BrowserUseManager will be unavailable.")


# Project Configuration
import config

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Content Validator ---
class ContentValidator:
    """Validates the adequacy of scraped content."""

    @staticmethod
    def is_content_adequate(clean_content_dict: Dict[str, str]) -> bool:
        """
        Checks if the extracted content meets basic requirements.
        Args:
            clean_content_dict: Dictionary with section keys and cleaned text content.
        Returns:
            True if content is adequate, False otherwise.
        """
        if not clean_content_dict:
            logger.warning("Content validation failed: No sections extracted.")
            return False

        # Check for presence of required sections
        missing_required = [
            section for section in config.REQUIRED_SECTIONS
            if section not in clean_content_dict or not clean_content_dict[section].strip()
        ]
        if missing_required:
            logger.warning(f"Content validation failed: Missing required sections: {missing_required}")
            # Allow proceeding if *some* content exists, but log the warning.
            # Return False here for stricter validation:
            # return False

        # Check total content length
        total_length = sum(len(content) for content in clean_content_dict.values())
        if total_length < config.MIN_CONTENT_LENGTH:
            logger.warning(f"Content validation failed: Total content length ({total_length}) is less than minimum ({config.MIN_CONTENT_LENGTH}).")
            return False

        logger.info("Content validation passed.")
        return True

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
            # --- MODIFICATION START ---
            # Prepare parameters based on config and Firecrawl API documentation
            params_to_pass = {}

            # Get page options from config
            page_options = config.FIRECRAWL_PARAMS.get('pageOptions', {})

            # Map config options to direct API parameters
            params_to_pass['onlyMainContent'] = page_options.get('onlyMainContent', False) # Default to False if not in config
            params_to_pass['screenshot'] = page_options.get('screenshot', False)
            if 'waitFor' in page_options: # Pass waitFor if defined
                 params_to_pass['waitFor'] = page_options['waitFor']

            # Determine 'formats' based on include flags
            formats = []
            if page_options.get('includeMarkdown', False): # Default to False if not set
                formats.append('markdown')
            if page_options.get('includeHtml', False):
                formats.append('html')
            # Add 'data' format if includeJson is True
            if page_options.get('includeJson', False):
                formats.append('data')

            # If no format specified, maybe default to markdown or html based on preference
            if not formats:
                formats.append('markdown') # Defaulting to markdown as per original config preference
                logger.debug("No specific format requested in config pageOptions, defaulting Firecrawl format to ['markdown']")

            params_to_pass['formats'] = formats

            # You can add other direct parameters from the API docs here if needed,
            # pulling values from config or using defaults. e.g.:
            # params_to_pass['timeout'] = config.FIRECRAWL_TIMEOUT * 1000 # API expects ms? Check docs. Defaulting to library's timeout arg.

            logger.debug(f"Calling Firecrawl scrape_url with URL: {url} and Params: {params_to_pass}")

            # Call the library function with unpacked parameters + url and timeout
            # NOTE: We pass timeout separately as it might be a direct argument to the library method
            #       and not part of the 'payload' parameters handled by **params_to_pass.
            scrape_result = self.app.scrape_url(
                url=url,
                timeout=config.FIRECRAWL_TIMEOUT,
                **params_to_pass # Unpack the prepared parameters
            )
            # --- MODIFICATION END ---


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
        
class BrowserUseManager:
    """
    Manages scraping using the browser-use framework.
    NOTE: This is a conceptual implementation. Actual usage depends heavily
    on the browser-use library's API. It might involve creating an Agent
    with a task description or using lower-level browser control functions.
    """
    def __init__(self):
        if not BROWSER_USE_AVAILABLE:
            logger.warning("BrowserUseManager initialized but library is unavailable.")
        else:
            # Placeholder for browser-use initialization if needed
            # e.g., setting up global config or a shared browser instance
            logger.info("BrowserUseManager initialized.")
            pass

    async def scrape(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrapes a URL using browser-use, attempting automated interactions.
        Args:
            url: The URL to scrape.
        Returns:
            A dictionary containing 'markdown' or 'html' content, or None on failure.
        """
        if not BROWSER_USE_AVAILABLE:
            logger.error("Cannot scrape with BrowserUseManager: browser-use library not available.")
            return None

        logger.info(f"Attempting browser-use scrape for: {url}")
        browser = None
        try:
            # --- This section needs to be adapted based on browser-use API ---
            # Option 1: Using an Agent (if applicable for generic scraping)
            # task_prompt = f"Extract the main academic profile content from {url}. Prioritize sections like publications, research, teaching. Handle cookie banners and load more buttons if necessary. Return the content as {config.BROWSER_USE_CONFIG['extraction_strategy']}."
            # agent = Agent(task=task_prompt, llm=...) # Might need LLM integration here too
            # history = await agent.run()
            # result = history.final_result() # Check format of result

            # Option 2: Using lower-level browser control (Simulated here)
            # This assumes browser-use provides a way to launch/connect to a browser
            # and perform configured interactions before extracting content.

            browser_config_options = {}
            if config.BROWSER_USE_CONFIG.get('use_proxy') and config.BROWSER_USE_CONFIG.get('proxy_url'):
                 # Proxy configuration syntax depends on browser-use library
                 # This is a guess:
                 browser_config_options['proxy'] = config.BROWSER_USE_CONFIG['proxy_url']
                 logger.info("Configuring browser-use with proxy.")

            # Example using Browser class directly (adjust based on actual API)
            browser_instance = Browser(config=BrowserConfig(**browser_config_options)) # Pass relevant config
            await browser_instance.open_tab(url=url)

            # --- Simulate Automated Interactions (Actual implementation depends on browser-use) ---
            logger.info("Attempting automated interactions (cookie consent, load more)...")
            # Example: Click cookie consent buttons
            for selector in config.BROWSER_USE_CONFIG['interaction_patterns'].get('cookie_consent', []):
                try:
                    # Assuming a method like click_if_exists exists
                    await browser_instance.click(selector, timeout=2) # Short timeout for non-essential clicks
                    logger.info(f"Clicked potential cookie consent: {selector}")
                except Exception: # Catch specific browser-use exceptions if possible
                    pass # Ignore if element not found

            # Example: Click load more buttons (might need multiple clicks)
            for selector in config.BROWSER_USE_CONFIG['interaction_patterns'].get('load_more', []):
                 max_clicks = 3 # Limit clicks to prevent infinite loops
                 for _ in range(max_clicks):
                    try:
                        # Assuming a method like click_if_exists exists
                        clicked = await browser_instance.click(selector, timeout=3) # Short timeout
                        if clicked:
                             logger.info(f"Clicked load more button: {selector}")
                             await asyncio.sleep(2) # Wait for content to load
                        else:
                            break # Stop if button not found or clickable
                    except Exception:
                        break # Stop on error or timeout


            # --- Simulate Content Extraction (Actual implementation depends on browser-use) ---
            logger.info(f"Extracting content using strategy: {config.BROWSER_USE_CONFIG['extraction_strategy']}")
            # Assuming a method like get_content exists
            content_data = await browser_instance.get_content(
                strategy=config.BROWSER_USE_CONFIG['extraction_strategy'],
                # Potentially pass element selectors for prioritization
                # selectors=config.BROWSER_USE_CONFIG['element_prioritization']
            )

            await browser_instance.close()
            browser = None # Ensure cleanup reference is cleared

            if content_data:
                # Assume content_data is a dict like {'markdown': '...', 'html': '...'} or just a string
                if isinstance(content_data, dict):
                     if config.BROWSER_USE_CONFIG['extraction_strategy'] in content_data and content_data[config.BROWSER_USE_CONFIG['extraction_strategy']]:
                        logger.info(f"browser-use scrape successful ({config.BROWSER_USE_CONFIG['extraction_strategy']}): {url}")
                        return {config.BROWSER_USE_CONFIG['extraction_strategy']: content_data[config.BROWSER_USE_CONFIG['extraction_strategy']]}
                     elif 'html' in content_data and content_data['html']: # Fallback to HTML
                         logger.info(f"browser-use scrape successful (HTML fallback): {url}")
                         return {'html': content_data['html']}
                     else:
                         logger.warning(f"browser-use returned data, but expected format ('{config.BROWSER_USE_CONFIG['extraction_strategy']}' or 'html') not found or empty.")
                         return None
                elif isinstance(content_data, str) and content_data.strip():
                    # If it returns just a string, assume it's HTML or Markdown based on strategy
                    logger.info(f"browser-use scrape successful (returned string): {url}")
                    return {config.BROWSER_USE_CONFIG['extraction_strategy']: content_data}
                else:
                    logger.warning(f"browser-use scrape returned empty or unexpected data type for {url}.")
                    return None
            else:
                logger.warning(f"browser-use scrape failed or returned no content for {url}.")
                return None
            # --- End of Adaptation Section ---

        except Exception as e:
            logger.error(f"browser-use error for {url}: {e}", exc_info=True)
            return None
        finally:
            if browser: # Ensure browser is closed if an error occurred mid-process
                try:
                    await browser.close()
                    logger.info("Closed browser-use instance after error.")
                except Exception as close_err:
                    logger.error(f"Error closing browser-use instance: {close_err}")


class PlaywrightManager:
    """Manages scraping using Playwright for fine-grained control."""

    async def scrape(self, url: str, specific_actions: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Scrapes a URL using Playwright, potentially performing specific actions.
        Args:
            url: The URL to scrape.
            specific_actions: Dictionary defining complex interactions if needed (not implemented in detail here).
        Returns:
            A dictionary containing 'html' content, or None on failure.
        """
        logger.info(f"Attempting Playwright scrape for: {url}")
        browser = None
        playwright = None
        try:
            playwright = await async_playwright().start()
            launch_options = {
                'headless': config.PLAYWRIGHT_HEADLESS,
                'timeout': config.PLAYWRIGHT_TIMEOUT,
            }
            if config.PLAYWRIGHT_USE_PROXY and config.PLAYWRIGHT_PROXY_CONFIG:
                launch_options['proxy'] = config.PLAYWRIGHT_PROXY_CONFIG
                logger.info("Launching Playwright with proxy.")

            browser = await playwright.chromium.launch(**launch_options)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                java_script_enabled=True,
                ignore_https_errors=True, # Be cautious with this in production
                viewport={'width': 1920, 'height': 1080} # Set a common viewport
            )
            page = await context.new_page()
            await page.goto(url, timeout=config.PLAYWRIGHT_TIMEOUT, wait_until='domcontentloaded') # Wait for DOM load

            # --- Add specific interactions here if needed ---
            # Example: Wait for a specific element crucial for content loading
            # try:
            #     await page.wait_for_selector('#main-content-area', timeout=15000)
            # except PlaywrightTimeoutError:
            #     logger.warning(f"Playwright: Timed out waiting for #main-content-area on {url}")

            # Example: Click cookie banners (more robustly than browser-use might)
            cookie_selectors = config.BROWSER_USE_CONFIG['interaction_patterns'].get('cookie_consent', [])
            for selector in cookie_selectors:
                 try:
                     await page.locator(selector).first.click(timeout=5000) # Click the first match
                     logger.info(f"Playwright clicked potential cookie consent: {selector}")
                     await page.wait_for_timeout(500) # Small delay after click
                 except PlaywrightTimeoutError:
                     pass # Element not found or visible within timeout
                 except Exception as click_err:
                     logger.warning(f"Playwright error clicking {selector}: {click_err}")


            # Example: Scroll down to trigger lazy loading
            logger.info("Playwright scrolling down page...")
            for _ in range(3): # Scroll down a few times
                 await page.evaluate('window.scrollBy(0, document.body.scrollHeight)')
                 await page.wait_for_timeout(1500) # Wait for content to potentially load


            # --- Get HTML content ---
            html_content = await page.content()

            await context.close()
            await browser.close()
            await playwright.stop()

            if html_content and html_content.strip():
                logger.info(f"Playwright scrape successful (HTML): {url}")
                return {'html': html_content}
            else:
                logger.warning(f"Playwright scrape returned empty HTML for {url}.")
                return None

        except PlaywrightTimeoutError:
             logger.error(f"Playwright timed out during operation for {url}.", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Playwright error for {url}: {e}", exc_info=True)
            return None
        finally:
            # Ensure resources are cleaned up
            if page and not page.is_closed():
                 await page.close()
            if context:
                 await context.close()
            if browser and browser.is_connected():
                 await browser.close()
            if playwright:
                 # Playwright doesn't have an is_active check, assume stop is safe
                 await playwright.stop()


# --- Orchestrator ---

class ScrapeOrchestrator:
    """Orchestrates the scraping process using multiple strategies."""

    def __init__(self):
        self.firecrawl_manager = FirecrawlManager(api_key=config.FIRE_CRAWL_API_KEY)
        self.browser_use_manager = BrowserUseManager()
        self.playwright_manager = PlaywrightManager()
        self.validator = ContentValidator()
        logger.info("ScrapeOrchestrator initialized.")

    async def get_content(self, url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Attempts to scrape content using Firecrawl -> BrowserUse -> Playwright fallback.
        Args:
            url: The URL to scrape.
        Returns:
            A tuple containing:
            - The raw scraped content (dict with 'markdown', 'html', or 'data') or None.
            - The name of the successful scraping tool ('firecrawl', 'browser-use', 'playwright') or None.
        """
        raw_output = None
        source_tool = None

        # 1. Try Firecrawl
        logger.info(f"Orchestrator: Starting with Firecrawl for {url}")
        raw_output = self.firecrawl_manager.scrape(url)
        if raw_output:
            source_tool = 'firecrawl'
            # Basic validation: Check if content exists before declaring success
            content_key = 'data' if 'data' in raw_output else ('markdown' if 'markdown' in raw_output else 'html')
            if raw_output.get(content_key):
                 logger.info(f"Orchestrator: Firecrawl succeeded for {url}.")
                 # Note: Full content adequacy check happens *after* section extraction in main.py
                 return raw_output, source_tool
            else:
                 logger.warning(f"Orchestrator: Firecrawl returned response but content for key '{content_key}' is empty. Proceeding to fallback.")
                 raw_output = None # Reset output to trigger fallback
                 source_tool = None


        # 2. Try Browser-Use (if Firecrawl failed or returned empty)
        if BROWSER_USE_AVAILABLE:
            logger.info(f"Orchestrator: Firecrawl failed or insufficient, trying browser-use for {url}")
            raw_output = await self.browser_use_manager.scrape(url)
            if raw_output:
                source_tool = 'browser-use'
                content_key = 'markdown' if 'markdown' in raw_output else 'html'
                if raw_output.get(content_key):
                    logger.info(f"Orchestrator: browser-use succeeded for {url}.")
                    return raw_output, source_tool
                else:
                     logger.warning(f"Orchestrator: browser-use returned response but content for key '{content_key}' is empty. Proceeding to fallback.")
                     raw_output = None
                     source_tool = None
        else:
            logger.warning("Orchestrator: Skipping browser-use as it's unavailable.")


        # 3. Try Playwright (if Firecrawl and BrowserUse failed or returned empty)
        logger.info(f"Orchestrator: Previous methods failed or insufficient, trying Playwright for {url}")
        raw_output = await self.playwright_manager.scrape(url)
        if raw_output:
            source_tool = 'playwright'
            if raw_output.get('html'):
                 logger.info(f"Orchestrator: Playwright succeeded for {url}.")
                 return raw_output, source_tool
            else:
                 logger.warning(f"Orchestrator: Playwright returned response but HTML content is empty.")
                 raw_output = None
                 source_tool = None


        # 4. If all fail
        if not raw_output:
            logger.error(f"Orchestrator: All scraping methods failed for {url}.")
            return None, None

        # This return should ideally not be reached if logic above is correct
        return raw_output, source_tool

