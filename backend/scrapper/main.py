# main.py
import argparse
import logging
import json
import os
import asyncio
import nest_asyncio # Handles asyncio event loop conflicts in some environments
import re # Using re for sanitizing filename
from datetime import datetime
from typing import Dict

# Apply nest_asyncio before importing other async libraries if needed
nest_asyncio.apply()

# Project Modules
import config
from scraping_manager import ScrapeOrchestrator
# The following imports are available but not directly used in this modified main.py
# as their logic is encapsulated within the ScrapeOrchestrator
# from processing import ContentProcessor
# from llm_handler import SchemaGenerator
# from scraping_manager import ContentValidator


# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Output Formatting ---
def save_output(data: Dict, output_dir: str, url: str, format: str = "json"):
    """Saves the final data to a file, naming it after the professor if possible."""
    filename = ""
    try:
        # Attempt to create a filename from the professor's name
        professor_name = data.get("profile_data", {}).get("personal_info", {}).get("full_name")
        if professor_name and isinstance(professor_name, str) and professor_name.strip():
            # Sanitize the name to be filesystem-friendly
            sane_name = professor_name.strip().lower().replace(" ", "_")
            sane_name = re.sub(r'(?u)[^-\w.]', '', sane_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{sane_name}_{timestamp}.{format}"
            logger.info(f"Generated filename from professor's name: {filename}")
        
        # Fallback to URL-based naming if the name is not available
        if not filename:
            logger.warning("Professor name not found in data, falling back to URL-based filename.")
            domain = url.split('/')[2].replace('.', '_')
            path_part = url.split('/')[-1] or url.split('/')[-2] # Get last part of path
            safe_path = "".join(c if c.isalnum() else "_" for c in path_part)[:50] # Sanitize
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{domain}_{safe_path}_{timestamp}.{format}"

    except Exception as e:
        logger.error(f"Error creating dynamic filename, using fallback. Error: {e}")
        # Final fallback in case of any unexpected error
        filename = f"profile_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"

    filepath = os.path.join(output_dir, filename)

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            if format == "json":
                json.dump(data, f, indent=4, ensure_ascii=False)
            else:
                 logger.error(f"Unsupported output format: {format}")
                 return None

        logger.info(f"Output successfully saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save output to {filepath}: {e}", exc_info=True)
        return None


# --- Main Execution ---
async def process_url(url: str, args: argparse.Namespace):
    """Processes a single URL through the scraping and structuring pipeline."""
    log_prefix = f"Processing {url} - "
    logger.info(f"{log_prefix}Starting...")

    # Initialize the main orchestrator which handles all sub-components
    orchestrator = ScrapeOrchestrator()

    final_output_data = None

    try:
        # --- Scrape and Process ---
        # The orchestrator now handles the entire pipeline:
        # scrape -> clean -> generate schema -> extract -> qc
        final_output_data = await orchestrator.get_and_process_content(url)

        if not final_output_data:
            logger.error(f"{log_prefix}Failed to retrieve and process content. No output generated.")
            return # Stop processing this URL

        logger.info(f"{log_prefix}Successfully processed and structured content.")
        logger.debug(f"{log_prefix}Final output keys: {list(final_output_data.keys())}")

        # --- Save the final output ---
        save_output(final_output_data, args.output, url, args.format)
        logger.info(f"{log_prefix}Processing finished successfully.")

    except Exception as e:
        logger.exception(f"{log_prefix}An unexpected error occurred during the main process: {e}")
        # Save an error state file
        error_metadata = {
            'source_url': url,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error_message': str(e),
            'schema_version_attempted': final_output_data.get("metadata", {}).get("schema_version") if final_output_data else None
        }
        save_output({'metadata': error_metadata}, config.OUTPUT_DIR, url, "json")


def main():
    parser = argparse.ArgumentParser(description="Precision Academic Profile Scraper")
    parser.add_argument("url", help="URL of the academic profile page to scrape.")
    parser.add_argument("-f", "--format", choices=["json"], default="json", # Extend choices later if needed
                        help="Output format (default: json)")
    parser.add_argument("-o", "--output", default=config.OUTPUT_DIR,
                        help=f"Output directory (default: {config.OUTPUT_DIR})")
    parser.add_argument("-s", "--schema-version",
                        help="Force the use of a specific schema version (e.g., 'v1' or '1'). Skips LLM schema generation.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Run the main processing function asynchronously
    try:
        asyncio.run(process_url(args.url, args))
    except Exception as e:
         logger.critical(f"Critical error running asyncio process: {e}", exc_info=True)

if __name__ == "__main__":
    main()