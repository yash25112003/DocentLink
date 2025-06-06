# main.py
import argparse
import logging
import json
import os
import asyncio
import nest_asyncio # Handles asyncio event loop conflicts in some environments
from datetime import datetime
from typing import Dict

# Apply nest_asyncio before importing other async libraries if needed
nest_asyncio.apply()

# Project Modules
import config
from scraping_manager import ScrapeOrchestrator, ContentValidator
from processing import ContentProcessor
from llm_handler import SchemaGenerator

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Output Formatting ---
def save_output(data: Dict, output_dir: str, url: str, format: str = "json"):
    """Saves the final data to a file."""
    # Create a filename based on URL (simplified)
    try:
        domain = url.split('/')[2].replace('.', '_')
        path_part = url.split('/')[-1] or url.split('/')[-2] # Get last part of path
        safe_path = "".join(c if c.isalnum() else "_" for c in path_part)[:50] # Sanitize
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{domain}_{safe_path}_{timestamp}.{format}"
    except Exception:
        filename = f"profile_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"

    filepath = os.path.join(output_dir, filename)

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            if format == "json":
                json.dump(data, f, indent=4, ensure_ascii=False)
            # Add CSV/Markdown formatters here if needed
            # elif format == "csv":
            #     # Requires flattening the data (e.g., using pandas)
            #     # df = pd.json_normalize(data) # Basic flattening
            #     # df.to_csv(filepath, index=False)
            #     logger.warning("CSV output not fully implemented.")
            #     f.write(json.dumps(data)) # Fallback to JSON string
            # elif format == "md":
            #     # Requires converting JSON to Markdown string
            #     logger.warning("Markdown output not fully implemented.")
            #     f.write(f"# Academic Profile Data\n\n```json\n{json.dumps(data, indent=2)}\n```") # Basic MD output
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

    # Initialize components
    orchestrator = ScrapeOrchestrator()
    content_processor = ContentProcessor()
    schema_generator = SchemaGenerator()
    validator = ContentValidator() # Use validator from scraping_manager

    final_output_data = None
    schema = None
    schema_version = None
    schema_docs = None

    try:
        # --- Phase 2: Scraping ---
        raw_output = await orchestrator.get_and_process_content(url)

        if not raw_output:
            logger.error(f"{log_prefix}Failed to retrieve and process content.")
            return # Stop processing this URL

        logger.info(f"{log_prefix}Successfully processed content.")
        logger.debug(f"{log_prefix}Raw output keys: {list(raw_output.keys())}")

        # Save the final output
        save_output(raw_output, args.output, url, args.format)
        logger.info(f"{log_prefix}Processing finished successfully.")

    except Exception as e:
        logger.exception(f"{log_prefix}An unexpected error occurred: {e}")
        # Save error state
        error_metadata = {
            'source_url': url,
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error_message': str(e),
            'schema_version_attempted': schema_version
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
