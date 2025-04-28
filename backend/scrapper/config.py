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
FIRECRAWL_PARAMS = {
    'pageOptions': {
        'onlyMainContent': False, # Get full page content initially
        'includeHtml': False,     # Prefer Markdown for easier section parsing
        'includeMarkdown': True,
        'includeJson': False,     # Set to True if JSON parsing is preferred over Markdown
        'screenshot': False,
        # 'waitFor': 3000, # Optional: wait time in ms
    },
    'crawlerOptions': {
        # 'includes': [], # Optional: Regex patterns for allowed URLs if crawling multiple pages
        # 'excludes': [], # Optional: Regex patterns for excluded URLs
        # 'maxDepth': 1, # Optional: Crawl depth
    }
}
FIRECRAWL_TIMEOUT = 120 # seconds

# Browser-Use Configuration
# Defines how browser-use should interact with pages.
# This is a simplified representation; actual implementation depends heavily
# on the browser-use library's specific API and capabilities.
BROWSER_USE_CONFIG = {
    'interaction_patterns': {
        # CSS selectors for elements to interact with automatically
        'cookie_consent': ['button#accept-cookies', '.cookie-banner .accept', '[id*="cookie"] button:contains("Accept")', '[class*="consent"] button'],
        'load_more': ['button.load-more', 'a.show-more', 'button:contains("Load More")', 'button:contains("Show More")'],
        'expand_sections': ['.expand-button', '.section-header button', '[aria-expanded="false"]'],
    },
    'element_prioritization': {
        # CSS selectors for key content sections to prioritize extraction
        # These might be used by browser-use or by our processing logic if browser-use returns full HTML
        'publications': ['.publications-list', 'div#publications', 'ul.pub-items', 'section[id*="publication"]'],
        'research_focus': ['.research-interests', 'section#research', 'div.research-areas', 'section[id*="research"]'],
        'teaching': ['.courses-taught', 'section#teaching', 'section[id*="teaching"]'],
        'biography': ['section#about', 'div.bio', 'section[id*="biography"]'],
        # Add more mappings based on SECTION_KEYWORDS below
    },
    'extraction_strategy': 'markdown', # Preferred output: 'markdown', 'html', or 'structured' (if supported)
    'timeout': 180, # seconds
    'use_proxy': bool(BRIGHTDATA_WSS_URL), # Flag to indicate if proxy should be used
    'proxy_url': BRIGHTDATA_WSS_URL
    # Add other browser-use specific configurations if needed
}

# Playwright Configuration
PLAYWRIGHT_TIMEOUT = 180 * 1000 # milliseconds
PLAYWRIGHT_HEADLESS = True # Run in headless mode
PLAYWRIGHT_USE_PROXY = bool(BRIGHTDATA_WSS_URL)
PLAYWRIGHT_PROXY_CONFIG = {
    "server": BRIGHTDATA_WSS_URL
} if PLAYWRIGHT_USE_PROXY else None


# --- Content Validation Thresholds ---
MIN_CONTENT_LENGTH = 150 # Minimum characters expected in combined *cleaned* sections
REQUIRED_SECTIONS = ['personal_background', 'research_focus', 'publications'] # Sections considered essential for adequacy

# --- Semantic Section Keywords ---
# Maps internal field names to potential variations found on web pages
SECTION_KEYWORDS = {
    'personal_background': ["biography", "about me", "cv", "curriculum vitae", "personal statement", "background", "profile", "about"],
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
    # Add more granular sections as needed
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
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

logger.info("Configuration loaded.")
