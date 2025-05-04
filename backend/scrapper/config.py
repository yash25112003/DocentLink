# config.py
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
FIRE_CRAWL_API_KEY = os.getenv("FIRE_CRAWL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BRIGHTDATA_WSS_URL = os.getenv("BRIGHTDATA_WSS_URL") # Optional proxy for Playwright/BrowserUse

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SCHEMA_HISTORY_DIR = os.path.join(BASE_DIR, 'schema_history')
LOG_FILE = os.path.join(BASE_DIR, 'scraper.log')

# --- LLM Configuration ---
LLM_MODEL_NAME = "gemini-1.5-flash" # Or "gemini-1.0-pro", "gemini-1.5-pro-latest" etc.
LLM_TEMPERATURE = 0.3
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 5 # seconds

# --- Scraping Tool Configurations ---

# Firecrawl Configuration
# Note: Parameters passed directly in scrape_url call now, this dict is for reference
FIRECRAWL_CONFIG_REFERENCE = {
    'pageOptions': {
        'onlyMainContent': False,
        'formats': ['markdown', 'html', 'data'],
        'waitFor': 3000
    },
    'crawlerOptions': {
        'maxDepth': 1
    }
}
FIRECRAWL_TIMEOUT = 150 # seconds

# Browser-Use Configuration
BROWSER_USE_CONFIG = {
    'interaction_patterns': {
        # *** CORRECTED SELECTORS for Playwright/Standard CSS ***
        'cookie_consent': [
            'button:has-text("Accept all")', # More specific first
            'button:has-text("Accept")',
            'button:has-text("Agree")',
            'button:has-text("Consent")',
            'button:has-text("Allow")',
            'button:has-text("OK")',
            'button:has-text("Okay")',
            'button:has-text("Got it")',
            'button:text-matches("accept", "i")', # Case-insensitive using Playwright regex
            '[id*="cookie"] button:text-matches("accept", "i")', # Case-insensitive
            '[class*="consent"] button:text-matches("accept", "i")', # Case-insensitive
            '[role="button"]:text-matches("accept", "i")',
            # Add other common patterns if needed
        ],
        'load_more': [
            'button:has-text("Load More")',
            'button:has-text("Show More")',
            'a:has-text("Load More")',
            'a:has-text("Show More")',
            'button:text-matches("load more", "i")',
            'button:text-matches("show more", "i")',
        ],
        'expand_sections': [
            '.expand-button',
            '.section-header button',
            '[aria-expanded="false"]',
            'button:has-text("Read more")'
        ],
    },
    'element_prioritization': {
        # CSS selectors for key content sections
        'publications': ['.publications-list', 'div#publications', 'ul.pub-items', 'section[id*="publication"]'],
        'research_focus': ['.research-interests', 'section#research', 'div.research-areas', 'section[id*="research"]'],
        'teaching': ['.courses-taught', 'section#teaching', 'section[id*="teaching"]'],
        'biography': ['section#about', 'div.bio', 'section[id*="biography"]', '#bio', '#about'],
    },
    'extraction_strategy': 'markdown', # Preferred output: 'markdown', 'html', or 'structured'
    'timeout': 180, # seconds
    'use_proxy': bool(BRIGHTDATA_WSS_URL),
    'proxy_url': BRIGHTDATA_WSS_URL,
    'disable_gif_creation': True
}

# Playwright Configuration
PLAYWRIGHT_TIMEOUT = 180 * 1000 # milliseconds (used for page navigation/actions)
PLAYWRIGHT_HEADLESS = True
PLAYWRIGHT_USE_PROXY = bool(BRIGHTDATA_WSS_URL)
PLAYWRIGHT_PROXY_CONFIG = {
    "server": BRIGHTDATA_WSS_URL
} if PLAYWRIGHT_USE_PROXY else None


# --- Content Validation Thresholds ---
MIN_CONTENT_LENGTH = 150 # Minimum characters expected in combined *cleaned* sections
# Sections that trigger a *warning* if missing, but don't stop processing if length > MIN_CONTENT_LENGTH
REQUIRED_SECTIONS = ['personal_background', 'research_focus', 'publications']

# --- Semantic Section Keywords ---
# (Keep this section as is)
SECTION_KEYWORDS = {
    'personal_background': ["biography", "about me", "cv", "curriculum vitae", "personal statement", "background", "profile", "about", "introduction"],
    'research_focus': ["research interests", "research area", "specializations", "expertise", "focus areas", "research", "scholarly interests"],
    'publications': ["publications", "papers", "journal articles", "conference proceedings", "peer-reviewed works", "publications list", "selected works", "bibliography", "academic writings", "articles", "presentations"],
    'current_research': ["ongoing research", "work in progress", "current studies", "active investigations", "current projects"],
    'research_projects': ["funded projects", "grants", "collaborative research", "project portfolio", "research funding", "sponsored research", "projects"],
    'books': ["books", "authored books", "monographs", "edited volumes", "textbooks", "book chapters"],
    'teaching': ["teaching", "courses taught", "teaching portfolio", "subjects", "lectures", "syllabi", "instruction", "courses"],
    'honors_awards': ["honors", "awards", "achievements", "prizes", "distinctions", "fellowships", "recognition"],
    'education': ["education", "academic history", "degrees", "alma mater", "academic background"],
    'professional_experience': ["experience", "positions", "work history", "employment", "career"],
    'contact_information': ["contact", "email", "phone", "address", "office location"],
    'links': ["links", "profiles", "external links", "google scholar", "orcid", "linkedin", "website", "social media"],
    'service': ["service", "committees", "professional service", "university service", "editorial boards"],
    'students_mentoring': ["students", "mentoring", "advisees", "lab members", "research group"],
}


# --- Logging Configuration ---
LOGGING_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': LOG_FILE,
    'filemode': 'a' # Append to log file
}

# --- Ensure Output Directories Exist ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCHEMA_HISTORY_DIR, exist_ok=True)

# --- Initialize Logging ---
# Basic config should be called once, preferably in main.py or the entry script
# logging.basicConfig(**LOGGING_CONFIG) # Comment out/remove if called elsewhere
logger = logging.getLogger(__name__) # Get logger instance

# Log loading confirmation - adjust if logging is set up in entry script
if not logging.getLogger().hasHandlers():
     logging.basicConfig(**LOGGING_CONFIG)
     logger.info("Config: Configuration loaded (logging initialized here).")
else:
    logger.info("Config: Configuration loaded.")