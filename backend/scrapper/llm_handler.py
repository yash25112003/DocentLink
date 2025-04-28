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
    # Depending on requirements, you might want to raise the error
    # or allow the application to continue without LLM features.
    # For now, we log the error and subsequent calls will fail.

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
            # Accessing the text content safely
            if response and response.candidates and response.candidates[0].content.parts:
                 return response.candidates[0].content.parts[0].text.strip()
            else:
                 # Log unexpected response structure
                 logger.warning(f"Gemini API returned unexpected response structure on attempt {attempt + 1}: {response}")
                 # You might want to inspect 'response.prompt_feedback' for blocked prompts
                 if response.prompt_feedback:
                      logger.warning(f"Prompt Feedback: {response.prompt_feedback}")
                 return None # Or raise an error after retries

        except Exception as e:
            logger.error(f"Gemini API error on attempt {attempt + 1}/{retries}: {e}", exc_info=True)
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Gemini API call failed after multiple retries.")
                return None
    return None # Should not be reached if retries > 0


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
                            # Store schema and documentation under version key
                            history[f"v{version}"] = schema_data # Use "vX" format for keys
                    except (ValueError, json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Could not load or parse schema file {filename}: {e}")
        except FileNotFoundError:
            logger.warning(f"Schema history directory not found: {self.history_dir}")
        except Exception as e:
            logger.error(f"Error loading schema history: {e}", exc_info=True)

        # Sort history by version number (descending) for easy access to latest
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

    def _build_schema_prompt(self, clean_content_dict: Dict[str, str], schema_history: Dict[str, Dict[str, Any]]) -> str:
        """Constructs the detailed prompt for the LLM schema analysis."""

        latest_schema_str = "No previous schemas available."
        latest_version = "N/A"
        if schema_history:
             # Get the latest version (first item in sorted dict)
             latest_version_key = next(iter(schema_history))
             latest_version = latest_version_key # e.g., "v3"
             latest_schema_data = schema_history[latest_version_key]
             latest_schema_str = f"Latest Schema (Version {latest_version}):\n```json\n{json.dumps(latest_schema_data.get('schema', {}), indent=2)}\n```\nDocumentation:\n{latest_schema_data.get('documentation', 'No documentation provided.')}"


        content_summary = "\n".join([f"- {key}: {value[:200]}..." if len(value) > 200 else f"- {key}: {value}"
                                     for key, value in clean_content_dict.items()])

        prompt = f"""
        Analyze the following extracted academic profile content and determine the most appropriate JSON schema structure.

        Extracted Content Summary:
        {content_summary}

        Schema History:
        {latest_schema_str}

        Task:
        1.  Carefully review the extracted content sections and their substance.
        2.  Compare the content against the latest schema version ({latest_version}) provided above.
        3.  Decision:
            a.  If the latest schema ({latest_version}) adequately represents the structure and key information in the *current* content, respond with the existing version number and a brief justification.
            b.  If the latest schema is *inadequate* (e.g., missing key fields present in the content, structure doesn't fit well), propose a NEW JSON schema structure. Assign it the next version number (e.g., if latest was v{latest_version}, the new one is v{int(latest_version[1:])+1 if latest_version != 'N/A' else 1}). Explain the *specific changes* made from the previous version and why they are necessary based on the current content.
        4.  Schema Requirements:
            * The schema should be a JSON object defining the structure.
            * Use clear, descriptive field names (snake_case).
            * Suggest appropriate data types (string, integer, boolean, list, object). Use 'list[string]' or 'list[object]' for lists. For objects within lists (like publications), define the nested structure.
            * Include fields for all major sections identified in the content summary.
            * Consider potential nesting (e.g., publications list within profile, education history list).
        5.  Output Format: Provide your response strictly in the following JSON format. Do NOT include any text outside this JSON structure.

            ```json
            {{
              "decision": "use_existing" | "propose_new",
              "schema_version": "vX", // The identified existing or proposed new version number (e.g., "v1", "v2")
              "justification": "Brief explanation for the decision (why existing fits, or why new is needed).",
              "proposed_schema": {{ ... JSON schema object ... }} | null, // Only include if decision is "propose_new"
              "schema_documentation": "Detailed description of the proposed schema fields, types, and purpose." | null // Only include if decision is "propose_new"
            }}
            ```

        Example `proposed_schema` structure:
        ```json
        {{
          "personal_info": {{
            "full_name": "string",
            "title": "string",
            "email": "string",
            "phone": "string",
            "office_location": "string"
          }},
          "biography": "string",
          "research_interests": "list[string]",
          "education": "list[object]", // Define object structure in documentation
          "publications": "list[object]" // Define object structure in documentation
          // ... other fields ...
        }}
        ```

        Analyze the content and provide the response in the specified JSON format ONLY.
        """
        return prompt


    def analyze_and_generate_schema(self, clean_content_dict: Dict[str, str]) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
        """
        Analyzes content, consults history, and uses LLM to get/generate a schema.
        Args:
            clean_content_dict: Dictionary of cleaned content sections.
        Returns:
            A tuple containing: (schema dictionary, schema version string, schema documentation string) or (None, None, None) on failure.
        """
        logger.info("Starting schema analysis and generation...")
        schema_history = self._load_schema_history()
        prompt = self._build_schema_prompt(clean_content_dict, schema_history)

        logger.debug(f"Schema generation prompt:\n{prompt[:500]}...") # Log beginning of prompt

        llm_response_str = _call_gemini_api(prompt)

        if not llm_response_str:
            logger.error("Failed to get response from LLM for schema generation.")
            return None, None, None

        logger.debug(f"Raw LLM schema response:\n{llm_response_str}")

        try:
            # Clean the response: Remove potential markdown code fences
            if llm_response_str.startswith("```json"):
                llm_response_str = llm_response_str[len("```json"):].strip()
            if llm_response_str.endswith("```"):
                llm_response_str = llm_response_str[:-len("```")].strip()

            response_data = json.loads(llm_response_str)

            decision = response_data.get("decision")
            schema_version = response_data.get("schema_version") # e.g., "v1"
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
                        logger.error(f"Existing schema version {schema_version} found in history but lacks schema or documentation.")
                        # Fallback: Try to generate a new one? Or fail? Let's try generating.
                        logger.warning("Falling back to generating a new schema as existing one is incomplete.")
                        # This requires re-prompting, potentially telling the LLM the chosen one was invalid.
                        # For simplicity now, we'll fail here.
                        return None, None, None
                else:
                    logger.error(f"LLM decided to use existing schema {schema_version}, but it was not found in history.")
                    # Fallback: Try to generate a new one?
                    logger.warning("Falling back to generating a new schema as chosen existing one not found.")
                    # Re-prompting needed. Fail for now.
                    return None, None, None

            elif decision == "propose_new":
                schema = response_data.get("proposed_schema")
                documentation = response_data.get("schema_documentation")
                if not schema or not documentation:
                     raise ValueError("LLM proposed new schema but 'proposed_schema' or 'schema_documentation' is missing.")

                try:
                    version_number = int(schema_version[1:]) # Extract number from "vX"
                    self._save_schema(version_number, schema, documentation)
                    logger.info(f"Successfully proposed and saved new schema version {schema_version}.")
                    return schema, schema_version, documentation
                except ValueError:
                     raise ValueError(f"Invalid schema version format from LLM: {schema_version}")
                except Exception as save_err:
                     logger.error(f"Failed to save proposed schema {schema_version}: {save_err}", exc_info=True)
                     # Return the proposed schema anyway, but log the save error
                     return schema, schema_version, documentation # Or fail completely: return None, None, None

            else:
                raise ValueError(f"Invalid 'decision' value from LLM: {decision}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from LLM: {e}\nRaw response was:\n{llm_response_str}", exc_info=True)
            return None, None, None
        except ValueError as e:
            logger.error(f"Invalid data in LLM response: {e}", exc_info=True)
            return None, None, None
        except Exception as e:
            logger.error(f"Error processing LLM schema response: {e}", exc_info=True)
            return None, None, None


# --- Data Extraction ---
class DataExtractor:
    """Uses LLM to extract structured data from text snippets based on a schema."""

    def _build_extraction_prompt(self, text_snippet: str, target_schema_fields: Dict) -> str:
        """Constructs the prompt for extracting specific fields from a text snippet."""

        schema_description = json.dumps(target_schema_fields, indent=2)

        prompt = f"""
        Extract information from the following text snippet according to the specified JSON schema fields.

        Text Snippet:
        ---
        {text_snippet}
        ---

        Target Schema Fields:
        ```json
        {schema_description}
        ```

        Task:
        1.  Read the text snippet carefully.
        2.  Identify the information corresponding to each field defined in the 'Target Schema Fields'.
        3.  Format the extracted information strictly as a JSON object matching the structure of 'Target Schema Fields'.
        4.  If information for a field is not found in the snippet, use `null` as the value for that field.
        5.  Ensure data types match the schema where possible (e.g., lists for list types). For lists of objects, extract each object.
        6.  Output ONLY the resulting JSON object. Do not include any explanations or text outside the JSON structure.

        Example Output Format:
        ```json
        {{
          "field_name_1": "extracted value",
          "field_name_2": ["item1", "item2"],
          "field_name_3": null,
          "list_of_objects_field": [
             {{ "nested_field_a": "value_a1", "nested_field_b": "value_b1" }},
             {{ "nested_field_a": "value_a2", "nested_field_b": null }}
          ]
          // ... other fields based on target schema ...
        }}
        ```

        Provide the extracted data in the specified JSON format ONLY.
        """
        return prompt

    def extract_structured_data(self, text_snippet: str, target_schema_fields: Dict) -> Optional[Dict]:
        """
        Uses LLM to extract structured data from a snippet.
        Args:
            text_snippet: The piece of text to extract from.
            target_schema_fields: A dictionary representing the desired fields and structure.
        Returns:
            A dictionary with the extracted data, or None on failure.
        """
        if not text_snippet or not target_schema_fields:
            logger.warning("Extraction skipped: Empty text snippet or target schema.")
            return None

        logger.info(f"Requesting LLM extraction for fields: {list(target_schema_fields.keys())}")
        prompt = self._build_extraction_prompt(text_snippet, target_schema_fields)
        logger.debug(f"LLM Extraction Prompt (Snippet Hash: {hash(text_snippet)}):\n{prompt[:500]}...")

        llm_response_str = _call_gemini_api(prompt)

        if not llm_response_str:
            logger.error("Failed to get response from LLM for data extraction.")
            return None

        logger.debug(f"Raw LLM extraction response:\n{llm_response_str}")

        try:
             # Clean the response: Remove potential markdown code fences
            if llm_response_str.startswith("```json"):
                llm_response_str = llm_response_str[len("```json"):].strip()
            if llm_response_str.endswith("```"):
                llm_response_str = llm_response_str[:-len("```")].strip()

            extracted_data = json.loads(llm_response_str)
            logger.info(f"Successfully extracted structured data using LLM.")
            return extracted_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON extraction response from LLM: {e}\nRaw response was:\n{llm_response_str}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error processing LLM extraction response: {e}", exc_info=True)
            return None

