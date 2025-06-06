# processing.py
import json
import logging
import re
import html
from typing import Dict, Any, Optional

from bs4 import BeautifulSoup

# Assumes these modules are in your project and configured correctly
import config
from llm_handler import DataExtractor, SchemaGenerator # Correctly import SchemaGenerator from llm_handler

# Setup logger for this module
logger = logging.getLogger(__name__)

class QualityControl:
    """Performs quality checks on structured data against its dynamically generated schema."""

    def run_quality_checks(self, structured_data: Dict[str, Any], schema: Dict) -> Dict[str, Any]:
        """
        Runs quality checks to verify the LLM's output against the schema it was given.
        The completeness score now reflects how well the LLM populated the schema it generated itself.
        """
        report = {
            "completeness_score": 0.0,
            "missing_fields": [],
            "notes": []
        }
        if not structured_data or not schema:
            report["notes"].append("Quality check failed: No structured data or schema provided.")
            return report

        # *** MODIFICATION START ***
        # The schema's actual fields are under the 'properties' key in a standard JSON schema.
        # If 'properties' doesn't exist, we fall back to the top-level keys of the schema itself.
        schema_fields = schema.get("properties", schema).keys()
        total_fields = len(schema_fields)

        if total_fields == 0:
            report["notes"].append("Quality check skipped: Schema has no fields to check.")
            return report

        filled_fields = 0
        missing_fields = []

        for field in schema_fields:
            data = structured_data.get(field)
            # A field is considered missing if it's null, an empty list, or an empty string.
            if data is None or (isinstance(data, list) and not data) or (isinstance(data, str) and not data.strip()):
                missing_fields.append(field)
            else:
                filled_fields += 1
        # *** MODIFICATION END ***

        report["completeness_score"] = round(filled_fields / total_fields, 2)
        report["missing_fields"] = missing_fields

        if report["completeness_score"] < 0.8:
            note = f"Data completeness score is low. The following fields were empty or not found in the text: {', '.join(missing_fields)}"
            report["notes"].append(note)
        else:
            report["notes"].append("Data completeness is high, indicating most schema fields were populated.")

        return report

class ContentProcessor:
    """Orchestrates the new, LLM-driven processing pipeline."""

    def __init__(self):
        # Use the SchemaGenerator from llm_handler which includes versioning
        self.schema_generator = SchemaGenerator()
        self.data_extractor = DataExtractor()
        self.quality_checker = QualityControl()
        logger.info("ContentProcessor initialized with dynamic, LLM-driven components.")

    def _clean_text(self, text: str) -> str:
        """
        Cleans and normalizes text for LLM processing by removing artifacts
        while preserving semantic structure like paragraphs.
        """
        if not text:
            return ""

        # Attempt to parse as HTML first to strip tags and get clean text
        try:
            soup = BeautifulSoup(text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
                tag.decompose()
            text = soup.get_text(separator='\n', strip=True)
        except Exception:
            pass

        text = html.unescape(text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'(\s*\n\s*){2,}', '\n\n', text).strip()
        text = re.sub(r'[ \t]+', ' ', text)

        logger.info("Text cleaning complete.")
        return text

    def process_and_structure(self, raw_content: Dict[str, Any], source_url: str) -> Optional[Dict[str, Any]]:
        """Main pipeline: clean content, generate schema, extract data, and perform QC."""
        logger.info(f"Starting new processing pipeline for URL: {source_url}")

        content_str = ""
        if isinstance(raw_content, dict):
            # Give preference to markdown, as it's often cleaner for LLMs
            content_str = raw_content.get('markdown') or raw_content.get('html') or json.dumps(raw_content.get('data'))
        elif isinstance(raw_content, str):
            content_str = raw_content

        if not content_str:
            logger.error("Raw content is empty or in an unsupported format.")
            return None

        clean_text = self._clean_text(content_str)

        min_len = getattr(config, 'MIN_CONTENT_LENGTH', 100)
        if len(clean_text) < min_len:
            logger.error(f"Cleaned text length ({len(clean_text)}) is below the minimum threshold of {min_len}.")
            return None

        # *** MODIFICATION START ***
        # The clean text should be passed as a dictionary for the schema generator in llm_handler
        clean_content_dict = {"main_content": clean_text}
        
        # Call the correct method from the versioning SchemaGenerator in llm_handler.py
        schema, schema_version, schema_doc = self.schema_generator.analyze_and_generate_schema(clean_content_dict)
        # *** MODIFICATION END ***
        
        if not schema:
            logger.error("Pipeline aborted: Failed to generate a dynamic schema.")
            return None

        structured_data = self.data_extractor.extract_structured_data(clean_text, schema)
        if not structured_data:
            logger.error("Pipeline aborted: Failed to extract structured data.")
            return None

        quality_report = self.quality_checker.run_quality_checks(structured_data, schema)
        logger.info(f"Final completeness score: {quality_report['completeness_score']}")

        final_output = {
            "metadata": {"source_url": source_url, "schema_version": schema_version},
            "schema_documentation": schema_doc,
            "quality_control": quality_report,
            "profile_data": structured_data
        }
        return final_output