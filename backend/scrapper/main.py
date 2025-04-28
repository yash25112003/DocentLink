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
from processing import ContentExtractor, DataStructurer, QualityControl
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
    content_extractor = ContentExtractor()
    schema_generator = SchemaGenerator()
    data_structurer = DataStructurer()
    quality_checker = QualityControl()
    validator = ContentValidator() # Use validator from scraping_manager

    final_output_data = None
    schema = None
    schema_version = None
    schema_docs = None

    try:
        # --- Phase 2: Scraping ---
        raw_output, source_tool = await orchestrator.get_content(url)

        if not raw_output or not source_tool:
            logger.error(f"{log_prefix}Failed to retrieve content using all methods.")
            return # Stop processing this URL

        logger.info(f"{log_prefix}Successfully scraped content using {source_tool}.")
        logger.debug(f"{log_prefix}Raw output keys: {list(raw_output.keys())}")


        # --- Phase 2/4: Section Identification & Initial Cleaning ---
        clean_content_dict = content_extractor.find_sections(raw_output, source_tool)

        if not clean_content_dict:
             logger.error(f"{log_prefix}Extracted content is empty after sectioning.")
             return # Stop processing this URL

        logger.info(f"{log_prefix}Identified {len(clean_content_dict)} potential key sections.")
        logger.debug(f"{log_prefix}Identified sections: {list(clean_content_dict.keys())}")

        # --- Phase 2 Validation (Post-Sectioning) ---
        if not validator.is_content_adequate(clean_content_dict):
             logger.error(f"{log_prefix}Extracted content failed adequacy validation after sectioning.")
             # Optionally save the inadequate raw/cleaned data for review
             # save_output({'metadata': {'source_url': url, 'status': 'inadequate_content'}, 'raw_output': raw_output, 'cleaned_sections': clean_content_dict}, config.OUTPUT_DIR, url, "json")
             return # Stop processing


        # --- Phase 3: Schema Design & Versioning ---
        if args.schema_version:
            # Load specific version if forced
            history = schema_generator._load_schema_history() # Use internal method for loading
            forced_version_key = f"v{args.schema_version}" if not args.schema_version.startswith('v') else args.schema_version
            if forced_version_key in history:
                schema_data = history[forced_version_key]
                schema = schema_data.get('schema')
                schema_docs = schema_data.get('documentation')
                schema_version = forced_version_key
                if not schema or not schema_docs:
                     logger.error(f"{log_prefix}Forced schema version {schema_version} loaded but is incomplete. Aborting.")
                     return
                logger.info(f"{log_prefix}Using forced schema version: {schema_version}")
            else:
                logger.error(f"{log_prefix}Forced schema version {forced_version_key} not found in history ({list(history.keys())}). Aborting.")
                return
        else:
            # Analyze content and generate/select schema
            schema, schema_version, schema_docs = schema_generator.analyze_and_generate_schema(clean_content_dict)
            if not schema or not schema_version or not schema_docs:
                 logger.error(f"{log_prefix}Failed to generate or select a schema. Aborting.")
                 return
            logger.info(f"{log_prefix}Generated/Selected schema version: {schema_version}")

        logger.debug(f"{log_prefix}Schema to be used:\n{json.dumps(schema, indent=2)}")


        # --- Phase 4: Structuring & QC ---
        structured_data = data_structurer.structure_data(clean_content_dict, schema)
        logger.info(f"{log_prefix}Structured data according to schema {schema_version}.")
        logger.debug(f"{log_prefix}Structured data sample:\n{json.dumps(structured_data, indent=2)[:500]}...")


        qc_report = quality_checker.run_quality_checks(structured_data, schema)
        logger.info(f"{log_prefix}Completed quality checks.")
        logger.debug(f"{log_prefix}QC Report:\n{json.dumps(qc_report, indent=2)}")


        # --- Phase 5: Output ---
        metadata = {
            'source_url': url,
            'timestamp': datetime.now().isoformat(),
            'scraping_tool_used': source_tool,
            'schema_version': schema_version,
            'status': 'success'
        }

        final_output_data = {
            'metadata': metadata,
            'schema_documentation': schema_docs,
            'quality_control': qc_report,
            'profile_data': structured_data
            # Optionally include clean_content_dict for debugging:
            # 'cleaned_content': clean_content_dict
        }

        # Save the final output
        save_output(final_output_data, args.output, url, args.format)
        logger.info(f"{log_prefix}Processing finished successfully.")

    except Exception as e:
        logger.exception(f"{log_prefix}An unexpected error occurred: {e}")
        # Optionally save error state
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
