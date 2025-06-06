# llm_handler.py
import google.generativeai as genai
import json
import os
import logging
import time
from typing import Dict, Any, Tuple, Optional, List

import config

# Setup logger for this module
logger = logging.getLogger(__name__)

# Configure the Gemini API client
try:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=config.GEMINI_API_KEY)
    logger.info("Gemini API configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}", exc_info=True)

# --- Helper Function for Robust API Calls ---
def _call_gemini_api(prompt: str, retries: int = config.LLM_MAX_RETRIES, delay: int = config.LLM_RETRY_DELAY) -> Optional[str]:
    """Calls the Gemini API with retry logic."""
    if not config.GEMINI_API_KEY:
         logger.error("Cannot call Gemini API: API key not configured.")
         return None

    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(config.LLM_MODEL_NAME)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.LLM_TEMPERATURE
                )
            )
            if response and response.candidates and response.candidates[0].content.parts:
                 return response.candidates[0].content.parts[0].text.strip()
            else:
                 logger.warning(f"Gemini API returned unexpected response structure on attempt {attempt + 1}: {response}")
                 if response.prompt_feedback:
                      logger.warning(f"Prompt Feedback: {response.prompt_feedback}")
                 return None

        except Exception as e:
            logger.error(f"Gemini API error on attempt {attempt + 1}/{retries}: {e}", exc_info=True)
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Gemini API call failed after multiple retries.")
                return None
    return None


# --- Schema Generation and Versioning ---
class SchemaGenerator:
    """Handles dynamic schema generation and versioning using an LLM."""

    def __init__(self, history_dir: str = config.SCHEMA_HISTORY_DIR):
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)
        logger.info(f"SchemaGenerator initialized. History directory: {self.history_dir}")

    def _load_schema_history(self) -> Dict[str, Dict[str, Any]]:
        """Loads all schema versions from the history directory."""
        history = {}
        try:
            for filename in os.listdir(self.history_dir):
                if filename.startswith("schema_v") and filename.endswith(".json"):
                    version_str = filename.replace("schema_v", "").replace(".json", "")
                    try:
                        version = int(version_str)
                        filepath = os.path.join(self.history_dir, filename)
                        with open(filepath, 'r') as f:
                            schema_data = json.load(f)
                            history[f"v{version}"] = schema_data
                    except (ValueError, json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not load or parse schema file {filename}: {e}")
        except Exception as e:
            logger.error(f"Error loading schema history: {e}", exc_info=True)

        sorted_history = dict(sorted(history.items(), key=lambda item: int(item[0][1:]), reverse=True))
        logger.info(f"Loaded {len(sorted_history)} schema versions from history.")
        return sorted_history

    def _save_schema(self, version: int, schema: Dict, documentation: str):
        """Saves a new schema version to the history directory."""
        filepath = os.path.join(self.history_dir, f"schema_v{version}.json")
        try:
            schema_data = {
                "version": f"v{version}",
                "schema": schema,
                "documentation": documentation
            }
            with open(filepath, 'w') as f:
                json.dump(schema_data, f, indent=4)
            logger.info(f"Saved new schema version v{version} to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save schema version v{version}: {e}", exc_info=True)

    # *** MODIFICATION START: More prescriptive schema generation prompt ***
    def _build_schema_prompt(self, clean_content_dict: Dict[str, str], schema_history: Dict[str, Dict[str, Any]]) -> str:
        """Constructs the detailed prompt for the LLM schema analysis."""
        latest_schema_str = "No previous schemas available."
        latest_version = "N/A"
        if schema_history:
             latest_version_key = next(iter(schema_history))
             latest_version = latest_version_key
             latest_schema_data = schema_history[latest_version_key]
             latest_schema_str = f"Latest Schema (Version {latest_version}):\n```json\n{json.dumps(latest_schema_data.get('schema', {}), indent=2)}\n```\nDocumentation:\n{latest_schema_data.get('documentation', 'No documentation provided.')}"

        content_summary = "\n".join([f"- {key}: {value[:200]}..." if len(value) > 200 else f"- {key}: {value}"
                                     for key, value in clean_content_dict.items()])
        prompt = f"""
        Analyze the following academic profile content and determine the most appropriate JSON schema.

        Extracted Content Summary:
        {content_summary}

        Schema History:
        {latest_schema_str}

        Task:
        1.  Review the content and compare it to the latest schema ({latest_version}).
        2.  Decide if the latest schema is adequate or if a new one is needed.
        3.  If a new schema is needed, propose one. Crucially, follow these structural rules:
            * **DO NOT** merge different types of information. Publications, patents, grants, and awards should be in separate top-level lists (e.g., `publications`, `patents`, `grants`, `awards_and_honors`).
            * **Personal Information:** The `personal_info` object should be comprehensive, containing `full_name`, `title`, `email`, `phone`, `office_location`, `department`, and `institution`.
            * **Education:** Create a dedicated `education` list of objects for degrees. Do not merge this information into other sections.
        4.  Output Format: Provide your response strictly in the following JSON format. Do NOT include any text outside this JSON structure.

            ```json
            {{
              "decision": "use_existing" | "propose_new",
              "schema_version": "vX",
              "justification": "Brief explanation for the decision.",
              "proposed_schema": {{ ... JSON schema object ... }} | null,
              "schema_documentation": "Detailed description of the proposed schema fields." | null
            }}
            ```

        Example of a **GOOD** `proposed_schema` structure to follow:
        ```json
        {{
          "personal_info": {{
            "full_name": "string",
            "title": "string",
            "email": "string",
            "department": "string",
            "institution": "string"
          }},
          "mission_statement": "string",
          "biography": "string",
          "research_interests": "list[string]",
          "education": "list[object]",
          "publications": "list[object]",
          "patents": "list[object]",
          "grants": "list[object]",
          "awards_and_honors": "list[object]",
          "affiliations": "list[string]"
        }}
        ```

        Analyze the content and provide the response in the specified JSON format ONLY.
        """
        return prompt
    # *** MODIFICATION END ***

    def analyze_and_generate_schema(self, clean_content_dict: Dict[str, str]) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
        """
        Analyzes content, consults history, and uses LLM to get/generate a schema.
        """
        logger.info("Starting schema analysis and generation...")
        schema_history = self._load_schema_history()
        prompt = self._build_schema_prompt(clean_content_dict, schema_history)

        logger.debug(f"Schema generation prompt:\n{prompt[:500]}...")
        llm_response_str = _call_gemini_api(prompt)

        if not llm_response_str:
            logger.error("Failed to get response from LLM for schema generation.")
            return None, None, None

        logger.debug(f"Raw LLM schema response:\n{llm_response_str}")
        try:
            if llm_response_str.startswith("```json"):
                llm_response_str = llm_response_str[len("```json"):].strip()
            if llm_response_str.endswith("```"):
                llm_response_str = llm_response_str[:-len("```")].strip()

            response_data = json.loads(llm_response_str)
            decision = response_data.get("decision")
            schema_version = response_data.get("schema_version")
            justification = response_data.get("justification")

            if not decision or not schema_version:
                 raise ValueError("LLM response missing 'decision' or 'schema_version'.")
            logger.info(f"LLM decision: {decision}, Version: {schema_version}, Justification: {justification}")

            if decision == "use_existing":
                if schema_version in schema_history:
                    existing_data = schema_history[schema_version]
                    schema = existing_data.get("schema")
                    documentation = existing_data.get("documentation")
                    if schema and documentation:
                        logger.info(f"Using existing schema version {schema_version}.")
                        return schema, schema_version, documentation
                    else:
                        logger.error(f"Existing schema {schema_version} is incomplete in history. Failing.")
                        return None, None, None
                else:
                    logger.error(f"LLM chose existing schema {schema_version}, but it was not found. Failing.")
                    return None, None, None
            elif decision == "propose_new":
                schema = response_data.get("proposed_schema")
                documentation = response_data.get("schema_documentation")
                if not schema or not documentation:
                     raise ValueError("LLM proposed new schema but 'proposed_schema' or 'schema_documentation' is missing.")
                try:
                    version_number = int(schema_version[1:])
                    self._save_schema(version_number, schema, documentation)
                    logger.info(f"Successfully proposed and saved new schema version {schema_version}.")
                    return schema, schema_version, documentation
                except ValueError:
                     raise ValueError(f"Invalid schema version format from LLM: {schema_version}")
            else:
                raise ValueError(f"Invalid 'decision' value from LLM: {decision}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error processing LLM schema response: {e}\nRaw response was:\n{llm_response_str}", exc_info=True)
            return None, None, None
        return None, None, None


# --- Data Extraction ---
class DataExtractor:
    """Uses LLM to extract structured data from text based on a dynamically generated schema."""

    # *** MODIFICATION START: More prescriptive extraction prompt ***
    def _generate_extraction_prompt(self, text_snippet: str, target_schema: Dict[str, Any]) -> str:
        """Generates a prompt for structured data extraction using a dynamic schema."""
        schema_str = json.dumps(target_schema, indent=2)
        prompt = f"""
        Objective: You are a meticulous data extraction specialist. Your task is to populate the given JSON schema using information from the provided source text from an academic's webpage.

        Source Text:
        ---
        {text_snippet[:8000]}
        ---

        Target JSON Schema:
        ---
        {schema_str}
        ---

        Extraction Rules:
        1.  **Strict Schema Adherence**: Use the exact field names and data structures defined in the schema.
        2.  **Categorize Correctly**: The 'Latest News' section contains mixed content. You MUST correctly identify and place items into the appropriate lists: `publications`, `patents`, `grants`, or `awards_and_honors`. Do NOT put patents or grants in the publications list.
        3.  **Comprehensive Extraction**: Extract ALL available information. For lists (like publications), extract every single item. For `personal_info`, populate all sub-fields like `title`, `department`, and `institution`.
        4.  **Handle Missing Information**: If information for a field is not in the text, use a `null` value. For lists that have no items, use an empty list `[]`.
        5.  **Clean Data**: Remove markdown, HTML, and unnecessary characters. Extract full dates (e.g., YYYY-MM-DD) when available.
        6.  **Final Output**: Your output must be a single, valid JSON object that perfectly matches the schema structure. Provide ONLY the final, populated JSON object as your response.

        Now, populate the schema based on the source text and the rules above.
        """
        return prompt
    # *** MODIFICATION END ***

    def extract_structured_data(self, text_snippet: str, target_schema: Dict) -> Optional[Dict]:
        """Uses LLM to extract structured data based on a dynamic schema."""
        if not text_snippet or not target_schema:
            logger.warning("Extraction skipped: Empty text snippet or target schema.")
            return None

        logger.info(f"Requesting LLM extraction for schema with top-level keys: {list(target_schema.keys())}")
        prompt = self._generate_extraction_prompt(text_snippet, target_schema)

        try:
            logger.debug(f"LLM Extraction Prompt:\n{prompt[:500]}...")
            response = _call_gemini_api(prompt, retries=3, delay=5)
            if not response:
                logger.error("LLM returned empty response for structured data extraction.")
                return None
            response = response.strip()
            if response.startswith("```json"):
                response = response[len("```json"):].strip()
            if response.endswith("```"):
                response = response[:-len("```")].strip()

            extracted_data = json.loads(response)
            logger.info("Successfully extracted and parsed structured data using LLM.")
            return extracted_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}\nRaw Response:\n{response}")
            return None
        except Exception as e:
            logger.error(f"Error during structured data extraction: {e}", exc_info=True)
            return None