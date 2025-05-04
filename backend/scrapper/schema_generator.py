import json
import os
import logging
from typing import Dict, Any, Tuple, Optional

import config
from llm_handler import _call_gemini_api

logger = logging.getLogger(__name__)

class SchemaGenerator:
    """Generates and manages schemas for academic content extraction."""
    
    def __init__(self):
        self.history_dir = config.SCHEMA_HISTORY_DIR
        os.makedirs(self.history_dir, exist_ok=True)
        logger.info(f"SchemaGenerator initialized. History directory: {self.history_dir}")

    def _load_schema_history(self) -> Dict[str, Dict]:
        """Loads all schema versions from the history directory."""
        schemas = {}
        try:
            for filename in os.listdir(self.history_dir):
                if filename.startswith('schema_v') and filename.endswith('.json'):
                    version = filename.split('_')[1].split('.')[0]
                    with open(os.path.join(self.history_dir, filename), 'r') as f:
                        schemas[version] = json.load(f)
            logger.info(f"Loaded {len(schemas)} schema versions from history.")
            return schemas
        except Exception as e:
            logger.error(f"Error loading schema history: {e}")
            return {}

    def _save_schema(self, version: str, schema: Dict) -> None:
        """Saves a schema version to the history directory."""
        try:
            filename = f"schema_{version}.json"
            with open(os.path.join(self.history_dir, filename), 'w') as f:
                json.dump(schema, f, indent=2)
            logger.info(f"Saved schema version {version} to {filename}")
        except Exception as e:
            logger.error(f"Error saving schema version {version}: {e}")

    def _generate_new_schema(self, content: Dict[str, str]) -> Tuple[Dict, str, str]:
        """Generates a new schema based on the content using LLM."""
        prompt = f"""
        Analyze the following academic content and generate a JSON schema that best represents its structure.
        The schema should be comprehensive enough to capture all relevant academic information.

        Content:
        {json.dumps(content, indent=2)}

        Requirements:
        1. The schema should include fields for all major academic sections (biography, research, publications, etc.)
        2. Each field should have appropriate type definitions and descriptions
        3. Include nested objects for complex sections (e.g., publications list)
        4. Make the schema flexible enough to handle variations in content structure

        Return a JSON object with:
        - schema: The generated JSON schema
        - version: A version string (e.g., "v18")
        - justification: A brief explanation of why this schema is appropriate

        Example format:
        {{
            "schema": {{
                "biography": {{
                    "type": "object",
                    "properties": {{
                        "markdown": {{ "type": "string" }},
                        "html": {{ "type": "string" }}
                    }}
                }},
                ...
            }},
            "version": "v18",
            "justification": "This schema captures the key academic sections..."
        }}
        """

        try:
            response = _call_gemini_api(prompt)
            if not response:
                raise ValueError("LLM returned empty response")
            
            # Clean and parse the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[len("```json"):].strip()
            if response.endswith("```"):
                response = response[:-len("```")].strip()
            
            result = json.loads(response)
            return result["schema"], result["version"], result["justification"]
        except Exception as e:
            logger.error(f"Error generating new schema: {e}")
            return {}, "v0", "Failed to generate schema"

    def analyze_and_generate_schema(self, content: Dict[str, str]) -> Tuple[Dict, str, str]:
        """Analyzes content and either uses an existing schema or generates a new one."""
        logger.info("Starting schema analysis and generation...")
        
        # Load existing schemas
        schemas = self._load_schema_history()
        if not schemas:
            logger.info("No existing schemas found. Generating new schema.")
            schema, version, justification = self._generate_new_schema(content)
            self._save_schema(version, schema)
            return schema, version, justification

        # Use LLM to decide whether to use existing schema or generate new one
        prompt = f"""
        Analyze the following content and existing schemas to determine if a new schema is needed.
        
        Content:
        {json.dumps(content, indent=2)}

        Existing Schema Versions:
        {json.dumps(list(schemas.keys()), indent=2)}

        Latest Schema (v{max(schemas.keys(), key=lambda x: int(x[1:]))}):
        {json.dumps(schemas[max(schemas.keys(), key=lambda x: int(x[1:]))], indent=2)}

        Decide whether to:
        1. Use the latest schema (return "use_latest")
        2. Generate a new schema (return "propose_new")

        If proposing new, also provide:
        - version: Next version number (e.g., "v18")
        - justification: Why a new schema is needed

        Return a JSON object with:
        {{
            "decision": "use_latest" or "propose_new",
            "version": "vX" (only if proposing new),
            "justification": "..." (only if proposing new)
        }}
        """

        try:
            response = _call_gemini_api(prompt)
            if not response:
                raise ValueError("LLM returned empty response")
            
            # Clean and parse the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[len("```json"):].strip()
            if response.endswith("```"):
                response = response[:-len("```")].strip()
            
            decision = json.loads(response)
            
            if decision["decision"] == "use_latest":
                latest_version = max(schemas.keys(), key=lambda x: int(x[1:]))
                logger.info(f"LLM decision: use_latest, Version: {latest_version}")
                return schemas[latest_version], latest_version, "Using latest schema"
            else:
                logger.info(f"LLM decision: propose_new, Version: {decision['version']}, Justification: {decision['justification']}")
                schema, version, justification = self._generate_new_schema(content)
                self._save_schema(version, schema)
                logger.info(f"Successfully proposed and saved new schema version {version}.")
                return schema, version, justification
                
        except Exception as e:
            logger.error(f"Error in schema analysis: {e}")
            # Fallback to latest schema
            latest_version = max(schemas.keys(), key=lambda x: int(x[1:]))
            return schemas[latest_version], latest_version, "Error in analysis, using latest schema" 