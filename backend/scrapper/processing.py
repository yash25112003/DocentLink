# processing.py
import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from bs4 import BeautifulSoup, NavigableString, Tag
import markdown # Python-Markdown library
from dateutil import parser as date_parser
from thefuzz import fuzz # For fuzzy matching section headers

import config
from llm_handler import DataExtractor # Import the LLM extractor

# Setup logger for this module
logger = logging.getLogger(__name__)

# Initialize LLM Data Extractor (optional, only if needed)
llm_data_extractor = DataExtractor()

# --- Content Extractor ---

class ContentExtractor:
    """Extracts and cleans content sections from raw scraped output."""

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if not text:
            return ""
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        # Add more specific cleaning rules if needed
        return text

    def _parse_markdown(self, markdown_text: str) -> Dict[str, str]:
        """Extracts sections from markdown text using heading-based parsing."""
        sections = {}
        current_section = None
        current_content = []
        
        # Split the markdown into lines
        lines = markdown_text.split('\n')
        
        for line in lines:
            # Check for headings (both ATX and Setext styles)
            if line.startswith('#') or (len(line) > 0 and all(c == '=' for c in line.strip()) or all(c == '-' for c in line.strip())):
                # If we have a current section, save it
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.strip('#=- ').lower()
                current_content = []
            else:
                # Add line to current section if we have one
                if current_section:
                    current_content.append(line)
        
        # Save the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # If no sections were found, try to extract content based on common academic profile patterns
        if not sections:
            # Look for common academic profile sections
            patterns = {
                'biography': r'(?i)(?:bio|biography|about me|about)(?:\s*:|\s*$)(.*?)(?=\n\n|\Z)',
                'research': r'(?i)(?:research interests|research focus|research areas)(?:\s*:|\s*$)(.*?)(?=\n\n|\Z)',
                'publications': r'(?i)(?:publications|selected publications|recent publications)(?:\s*:|\s*$)(.*?)(?=\n\n|\Z)',
                'teaching': r'(?i)(?:teaching|courses|education)(?:\s*:|\s*$)(.*?)(?=\n\n|\Z)',
                'service': r'(?i)(?:service|professional service|academic service)(?:\s*:|\s*$)(.*?)(?=\n\n|\Z)',
                'awards': r'(?i)(?:awards|honors|achievements)(?:\s*:|\s*$)(.*?)(?=\n\n|\Z)'
            }
            
            for section, pattern in patterns.items():
                match = re.search(pattern, markdown_text, re.DOTALL)
                if match:
                    sections[section] = match.group(1).strip()
        
        # If still no sections, try to extract content between major headings
        if not sections:
            # Split by major headings (lines with multiple = or -)
            parts = re.split(r'\n[-=]{3,}\n', markdown_text)
            if len(parts) > 1:
                for i, part in enumerate(parts):
                    if i == 0:
                        sections['introduction'] = part.strip()
                    else:
                        # Try to extract a section name from the first line
                        first_line = part.split('\n')[0].strip()
                        section_name = first_line.lower().replace(' ', '_')
                        sections[section_name] = part.strip()
        
        # Clean up the sections
        cleaned_sections = {}
        for section, content in sections.items():
            if content.strip():
                cleaned_sections[section] = self._clean_text(content)
        
        logger.info(f"Extracted {len(cleaned_sections)} sections from markdown.")
        return cleaned_sections

    def _parse_html(self, html_content: str) -> Dict[str, str]:
        """Parses HTML content to extract sections, potentially using hints."""
        sections = {}
        try:
            soup = BeautifulSoup(html_content, 'lxml')

            # Attempt to find sections based on common structures or config hints
            # Strategy 1: Look for <section> tags or divs with relevant IDs/classes
            potential_sections = soup.find_all(['section', 'div'], id=re.compile('|'.join(config.SECTION_KEYWORDS.keys()), re.I))
            potential_sections += soup.find_all(['section', 'div'], class_=re.compile('|'.join(config.SECTION_KEYWORDS.keys()), re.I))

            # Strategy 2: Look for headers (h2, h3) and grab subsequent content
            headers = soup.find_all(['h2', 'h3', 'h4']) # Prioritize common section headers

            processed_elements = set() # Keep track of elements assigned to a section

            # Process headers first
            for header in headers:
                 header_text = self._clean_text(header.get_text())
                 if not header_text or header in processed_elements:
                      continue

                 content = []
                 # Find siblings until the next header or a limiting tag
                 for sibling in header.find_next_siblings():
                      if sibling.name.startswith('h') or sibling.name in ['section', 'footer', 'nav']: # Stop at next header or major section break
                           break
                      if sibling not in processed_elements:
                           # Check if sibling is a container with relevant info
                           if isinstance(sibling, Tag):
                                # Avoid adding empty containers or script tags
                                if sibling.name == 'script' or not sibling.get_text(strip=True):
                                     continue
                                content.append(sibling.get_text(separator=' ', strip=True))
                                processed_elements.add(sibling) # Mark as processed
                           elif isinstance(sibling, NavigableString) and sibling.strip():
                                content.append(sibling.strip())
                                # Cannot add NavigableString to set, but its content is captured

                 if header_text and content:
                      sections[header_text] = self._clean_text(" ".join(content))
                      processed_elements.add(header)


            # Add content from specifically identified sections (Strategy 1) if not already captured
            for section_tag in potential_sections:
                 if section_tag in processed_elements:
                      continue
                 # Try to find a representative header within the section or use ID/class
                 section_header_tag = section_tag.find(['h2', 'h3', 'h4'])
                 section_key = self._clean_text(section_header_tag.get_text()) if section_header_tag else section_tag.get('id') or (section_tag.get('class') and section_tag.get('class')[0])
                 section_key = section_key or f"section_{len(sections)}" # Fallback key

                 section_content = self._clean_text(section_tag.get_text(separator=' ', strip=True))

                 if section_key and section_content and section_key not in sections:
                      sections[section_key] = section_content
                      processed_elements.add(section_tag) # Add container to processed


            # Basic cleanup if no sections found (e.g., grab main content area)
            if not sections:
                 main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main') or soup.body
                 if main_content:
                      sections['main_content'] = self._clean_text(main_content.get_text(separator=' ', strip=True))

        except Exception as e:
            logger.error(f"Error parsing HTML content: {e}", exc_info=True)

        logger.info(f"Extracted {len(sections)} potential sections from HTML.")
        return sections

    def _parse_json_data(self, data: Any) -> Dict[str, str]:
        """Flattens and cleans structured JSON data into a dict for keyword matching."""
        sections = {}
        
        def flatten(json_obj, prefix=''):
            if isinstance(json_obj, dict):
                for k, v in json_obj.items():
                    new_key = f"{prefix}_{k}" if prefix else k
                    if isinstance(v, (str, int, float, bool)):
                        sections[new_key] = self._clean_text(str(v))
                    elif isinstance(v, list):
                        if all(isinstance(item, (str, int, float, bool)) for item in v):
                            sections[new_key] = self._clean_text(", ".join(map(str, v)))
                        elif all(isinstance(item, dict) for item in v):
                            # Handle list of objects (like publications)
                            for i, item in enumerate(v):
                                flatten(item, f"{new_key}_{i}")
                        else:
                            sections[new_key] = self._clean_text(", ".join(map(str, v)))
                    else:
                        flatten(v, new_key)
            elif isinstance(json_obj, list):
                if all(isinstance(item, (str, int, float, bool)) for item in json_obj):
                    sections[prefix] = self._clean_text(", ".join(map(str, json_obj)))
                elif all(isinstance(item, dict) for item in json_obj):
                    for i, item in enumerate(json_obj):
                        flatten(item, f"{prefix}_{i}")
                else:
                    sections[prefix] = self._clean_text(", ".join(map(str, json_obj)))
            elif isinstance(json_obj, (str, int, float, bool)):
                sections[prefix] = self._clean_text(str(json_obj))

        flatten(data)
        logger.info(f"Processed {len(sections)} key-value pairs from JSON data.")
        return sections


    def _match_sections(self, extracted_sections: Dict[str, str]) -> Dict[str, str]:
        """Matches extracted section headers/keys to predefined SECTION_KEYWORDS."""
        matched_content = {}
        used_keys = set()

        # Academic profile specific section mappings
        academic_mappings = {
            'biography': ['bio', 'about', 'about me', 'introduction', 'background'],
            'research': ['research', 'research interests', 'research focus', 'research areas', 'interests'],
            'publications': ['publications', 'selected publications', 'recent publications', 'papers', 'research papers'],
            'teaching': ['teaching', 'courses', 'education', 'teaching experience'],
            'service': ['service', 'professional service', 'academic service', 'committee service'],
            'awards': ['awards', 'honors', 'achievements', 'grants', 'funding'],
            'education': ['education', 'academic background', 'degrees', 'phd', 'ph.d.'],
            'experience': ['experience', 'work experience', 'professional experience', 'positions'],
            'contact': ['contact', 'contact information', 'email', 'address'],
            'projects': ['projects', 'research projects', 'current projects', 'ongoing projects']
        }

        # First pass: exact and close matches
        for target_key, keywords in academic_mappings.items():
            best_match_score = 0
            best_match_content = ""
            best_match_source_key = None

            for source_key, source_content in extracted_sections.items():
                if not source_content or source_key in used_keys:
                    continue

                # Convert to lowercase for comparison
                source_key_lower = source_key.lower()
                
                # Check for exact matches first
                if source_key_lower in keywords:
                    best_match_score = 100
                    best_match_content = source_content
                    best_match_source_key = source_key
                    break

                # Check fuzzy match score against all keywords
                current_best_score = 0
                for keyword in keywords:
                    # Use token_set_ratio for flexibility with word order and partial matches
                    score = fuzz.token_set_ratio(keyword, source_key_lower)
                    current_best_score = max(current_best_score, score)

                if current_best_score > best_match_score:
                    best_match_score = current_best_score
                    best_match_content = source_content
                    best_match_source_key = source_key

            # If a good enough match is found
            MATCH_THRESHOLD = 75
            if best_match_score >= MATCH_THRESHOLD and best_match_source_key:
                matched_content[target_key] = best_match_content
                used_keys.add(best_match_source_key)
                logger.debug(f"Matched '{best_match_source_key}' to '{target_key}' (Score: {best_match_score})")

        # Second pass: content-based matching for unmatched sections
        for source_key, source_content in extracted_sections.items():
            if source_key in used_keys:
                continue

            # Look for keywords in the content itself
            content_lower = source_content.lower()
            for target_key, keywords in academic_mappings.items():
                if target_key in matched_content:
                    continue

                for keyword in keywords:
                    if keyword in content_lower:
                        matched_content[target_key] = source_content
                        used_keys.add(source_key)
                        logger.debug(f"Content-based match: '{source_key}' to '{target_key}'")
                        break

        # Third pass: handle special cases
        for source_key, source_content in extracted_sections.items():
            if source_key in used_keys:
                continue

            # Handle publications with specific formats
            if any(venue in source_content for venue in ['ACM', 'IEEE', 'USENIX', 'CCS', 'S&P', 'NDSS']):
                if 'publications' not in matched_content:
                    matched_content['publications'] = source_content
                    used_keys.add(source_key)
                    logger.debug(f"Venue-based match: '{source_key}' to 'publications'")

            # Handle education with degree indicators
            elif any(degree in source_content for degree in ['PhD', 'Ph.D.', 'M.S.', 'M.Sc.', 'B.S.', 'B.Sc.']):
                if 'education' not in matched_content:
                    matched_content['education'] = source_content
                    used_keys.add(source_key)
                    logger.debug(f"Degree-based match: '{source_key}' to 'education'")

        # Add any remaining unmatched sections with generic keys
        for source_key, source_content in extracted_sections.items():
            if source_key not in used_keys and source_content.strip():
                generic_key = f"additional_{source_key.lower().replace(' ', '_')}"
                matched_content[generic_key] = source_content
                logger.debug(f"Added unmatched section as '{generic_key}'")

        logger.info(f"Mapped extracted content to {len(matched_content)} target sections.")
        return matched_content


    def find_sections(self, raw_content: str, source_tool: str = "browser-use") -> Dict[str, str]:
        """Extracts sections from raw content based on the source tool."""
        sections = {}
        
        # Handle browser-use output directly
        if source_tool == "browser-use":
            try:
                # Try to parse as JSON first
                if isinstance(raw_content, dict):
                    content_dict = raw_content
                else:
                    content_dict = json.loads(raw_content)
                
                # Process main content
                if 'markdown' in content_dict:
                    sections.update(self._parse_markdown(content_dict['markdown']))
                elif 'html' in content_dict:
                    sections.update(self._parse_html(content_dict['html']))
                elif 'data' in content_dict:
                    sections.update(self._parse_json_data(content_dict['data']))
                
                # Process additional content from linked pages
                if 'additional_content' in content_dict:
                    for key, content in content_dict['additional_content'].items():
                        try:
                            # Try to parse as JSON first
                            if isinstance(content, dict):
                                additional_sections = self._parse_json_data(content)
                            else:
                                # Try to parse as markdown
                                additional_sections = self._parse_markdown(content)
                            
                            # Merge with existing sections
                            for section_key, section_content in additional_sections.items():
                                if section_key in sections:
                                    # Append content if section already exists
                                    sections[section_key] += "\n\n" + section_content
                                else:
                                    sections[section_key] = section_content
                            
                            logger.info(f"Successfully processed additional content from {key}")
                        except Exception as e:
                            logger.warning(f"Failed to process additional content from {key}: {e}")
                
                return sections
            except json.JSONDecodeError:
                # If not JSON, try parsing as markdown
                return self._parse_markdown(raw_content)
            except Exception as e:
                logger.error(f"Error processing browser-use output: {e}")
                return {}
        
        # Handle other tools (Firecrawl, etc.)
        try:
            if source_tool == "firecrawl":
                return self._parse_markdown(raw_content)
            else:
                logger.warning(f"Unknown source tool: {source_tool}")
                return {}
        except Exception as e:
            logger.error(f"Error in find_sections: {e}")
            return {}


# --- Data Structurer ---

class DataStructurer:
    """Structures the cleaned content according to the provided schema."""

    def _parse_date(self, date_string: str) -> Optional[str]:
        """Attempts to parse a date string into ISO format."""
        if not date_string:
            return None
        try:
            # fuzzy=True helps with slightly malformed dates within text
            dt = date_parser.parse(date_string, fuzzy=True)
            return dt.date().isoformat() # Return only date part in ISO format
        except (ValueError, OverflowError, TypeError):
            logger.debug(f"Could not parse date: {date_string}")
            return None # Return None if parsing fails

    def _structure_list_items(self, text_block: str, item_schema: Dict) -> List[Dict]:
        """Uses LLM to structure items from a text block (e.g., publications list)."""
        if not text_block or not item_schema:
            return []
        logger.info("Using LLM to structure list items...")

        # Construct a simplified schema prompt for list items
        list_prompt = f"""
        The following text block contains a list of items (e.g., publications, positions, degrees).
        Extract each distinct item and structure it according to the provided JSON schema for a single item.
        Return a JSON list containing the structured objects for each item found.

        Text Block:
        ---
        {text_block[:2000]}...
        ---

        Schema for ONE item:
        ```json
        {json.dumps(item_schema, indent=2)}
        ```

        Task:
        1. Identify individual items in the text block.
        2. For each item, extract information matching the fields in the item schema.
        3. Use `null` for missing fields within an item.
        4. Return ONLY a JSON list `[...]` where each element is a JSON object structured like the item schema.
           If no items are found or extraction fails, return an empty list `[]`.

        Example Output:
        ```json
        [
          {{ "title": "...", "authors": [...], "year": ... }},
          {{ "title": "...", "authors": [...], "year": ... }}
        ]
        ```

        Provide the JSON list ONLY.
        """
        from llm_handler import _call_gemini_api # Use the helper directly
        llm_response_str = _call_gemini_api(list_prompt)

        if not llm_response_str:
             logger.error("LLM failed to structure list items.")
             return []

        try:
            # Clean the response
            if llm_response_str.startswith("```json"):
                llm_response_str = llm_response_str[len("```json"):].strip()
            if llm_response_str.endswith("```"):
                llm_response_str = llm_response_str[:-len("```")].strip()

            # Ensure the response is actually a list
            parsed_list = json.loads(llm_response_str)
            if isinstance(parsed_list, list):
                 logger.info(f"LLM successfully structured {len(parsed_list)} list items.")
                 # Optional: Add date parsing for specific fields within each item here
                 for item in parsed_list:
                      if 'year' in item and isinstance(item['year'], (str, int)):
                           # Try parsing year as date if needed, or just keep as is
                           pass
                      if 'date' in item and isinstance(item['date'], str):
                           item['date'] = self._parse_date(item['date']) or item['date'] # Keep original if parse fails

                 return parsed_list
            else:
                 logger.warning(f"LLM response for list items was not a JSON list: {llm_response_str}")
                 return []

        except json.JSONDecodeError:
            logger.error(f"Failed to decode LLM response for list items: {llm_response_str}")
            return []
        except Exception as e:
            logger.error(f"Error processing LLM list item response: {e}", exc_info=True)
            return []


    def structure_data(self, clean_content_dict: Dict[str, str], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Populates a structured dictionary based on the schema and cleaned content.
        Args:
            clean_content_dict: Dictionary mapping section keywords to cleaned text.
            schema: The target JSON schema definition.
        Returns:
            A nested dictionary containing the structured data.
        """
        structured_data = {}
        logger.info("Structuring data according to schema...")

        # Iterate through the top-level fields defined in the schema
        for field, field_schema in schema.items():
            field_type = field_schema if isinstance(field_schema, str) else field_schema.get('type', 'string') # Simple schema: value is type; Complex: dict with 'type'
            field_content = clean_content_dict.get(field) # Get content matched earlier

            if not field_content:
                structured_data[field] = None # Or [] for list types? Schema should define default/required
                continue

            try:
                if field_type == 'string':
                    structured_data[field] = field_content
                elif field_type == 'integer':
                    # Attempt to extract first integer found
                    match = re.search(r'\d+', field_content)
                    structured_data[field] = int(match.group(0)) if match else None
                elif field_type == 'boolean':
                    # Simple check for truthiness (presence of content)
                    structured_data[field] = bool(field_content)
                elif field_type == 'date':
                     structured_data[field] = self._parse_date(field_content)
                elif field_type.startswith('list'):
                    # Handle lists (e.g., list[string], list[object])
                    if field_type == 'list[string]':
                        # Simple split by newline or common delimiters, then clean
                        items = re.split(r'\n|,|;', field_content)
                        structured_data[field] = [item.strip() for item in items if item.strip()]
                        # Could potentially use LLM for smarter list splitting/cleaning here too
                    elif field_type == 'list[object]':
                        # This requires knowing the schema of the objects within the list
                        # Assume the schema definition includes item structure, e.g.,
                        # "publications": {"type": "list[object]", "items": {"title": "string", "year": "integer"}}
                        item_schema = field_schema.get('items') if isinstance(field_schema, dict) else None
                        if item_schema:
                            # Use LLM to parse the block of text into structured list items
                            structured_data[field] = self._structure_list_items(field_content, item_schema)
                        else:
                            logger.warning(f"Schema for list field '{field}' is missing 'items' definition. Storing raw text.")
                            # Fallback: store raw text or attempt simple split
                            items = re.split(r'\n\n|\n\s*\n', field_content) # Split by double newline first
                            if len(items) <= 1:
                                 items = re.split(r'\n', field_content) # Split by single newline
                            structured_data[field] = [item.strip() for item in items if item.strip()]
                    else: # Generic list or unknown list type
                         items = re.split(r'\n|,|;', field_content)
                         structured_data[field] = [item.strip() for item in items if item.strip()]

                elif field_type == 'object':
                    # Handle nested objects - requires schema definition for the object
                    # Similar to list[object], needs the nested schema structure
                    nested_schema = field_schema.get('properties') if isinstance(field_schema, dict) else None
                    if nested_schema:
                         logger.info(f"Using LLM to structure nested object field: {field}")
                         # Use LLM to extract structure from the field's content
                         structured_data[field] = llm_data_extractor.extract_structured_data(field_content, nested_schema)
                    else:
                         logger.warning(f"Schema for object field '{field}' is missing 'properties' definition. Storing raw text.")
                         structured_data[field] = field_content # Fallback
                else:
                    # Default to string if type is unknown or complex
                    structured_data[field] = field_content

            except Exception as e:
                logger.error(f"Error structuring field '{field}' (Type: {field_type}): {e}. Storing raw content.", exc_info=True)
                structured_data[field] = field_content # Store raw content on error

        # --- Post-processing (Example: Temporal Sequencing) ---
        # This requires specific schema fields like 'publications' with 'year' or 'date'
        # Example: Sort publications by year if present
        if 'publications' in structured_data and isinstance(structured_data['publications'], list):
            try:
                # Sort assuming 'year' or 'date' field exists within each publication object
                structured_data['publications'].sort(
                    key=lambda x: x.get('year') or self._parse_date(x.get('date', '') or ''),
                    reverse=True
                )
                logger.info("Sorted publications chronologically (descending).")
            except Exception as sort_err:
                logger.warning(f"Could not sort publications: {sort_err}")


        logger.info("Data structuring complete.")
        return structured_data


# --- Quality Control ---

class QualityControl:
    """Performs quality checks on the structured data."""

    def _check_completeness(self, structured_data: Dict, schema: Dict) -> Tuple[float, List[str]]:
        """Checks for presence of required fields based on schema (basic check)."""
        total_fields = len(schema)
        present_fields = 0
        missing_fields = []

        if total_fields == 0:
            return 1.0, []

        for field, field_schema in schema.items():
            is_required = field_schema.get('required', False) if isinstance(field_schema, dict) else False # Assume optional if not specified
            # Check if field exists and has a non-null/non-empty value
            if field in structured_data and structured_data[field] not in [None, "", []]:
                 present_fields += 1
            elif is_required:
                 missing_fields.append(field)


        completeness_score = present_fields / total_fields if total_fields > 0 else 1.0
        logger.info(f"Completeness check: {present_fields}/{total_fields} fields populated. Missing required: {missing_fields}")
        return round(completeness_score, 2), missing_fields

    def _detect_duplicates(self, structured_data: Dict, schema: Dict) -> Dict[str, List[Any]]:
        """Detects potential duplicates within list fields."""
        duplicates = {}
        DUPLICATE_THRESHOLD = 90 # Fuzzy match score threshold

        for field, field_schema in schema.items():
             field_type = field_schema if isinstance(field_schema, str) else field_schema.get('type', 'string')
             if field_type.startswith('list') and field in structured_data and isinstance(structured_data[field], list):
                  items = structured_data[field]
                  if not items or len(items) < 2:
                       continue

                  potential_duplicates_indices = set()
                  # Compare each item with subsequent items
                  for i in range(len(items)):
                       for j in range(i + 1, len(items)):
                            item1_str = json.dumps(items[i]) if isinstance(items[i], dict) else str(items[i])
                            item2_str = json.dumps(items[j]) if isinstance(items[j], dict) else str(items[j])

                            # Use fuzzy matching on string representations
                            score = fuzz.ratio(item1_str, item2_str)
                            if score >= DUPLICATE_THRESHOLD:
                                 potential_duplicates_indices.add(i)
                                 potential_duplicates_indices.add(j)
                                 logger.debug(f"Potential duplicate found in '{field}' (Score: {score}): {item1_str} vs {item2_str}")


                  if potential_duplicates_indices:
                       duplicates[field] = [items[idx] for idx in sorted(list(potential_duplicates_indices))]

        if duplicates:
             logger.warning(f"Potential duplicates detected in fields: {list(duplicates.keys())}")
        else:
             logger.info("No significant duplicates detected in list fields.")
        return duplicates

    def _annotate_temporal_relevance(self, structured_data: Dict, schema: Dict) -> Dict:
        """Adds simple temporal flags (e.g., 'recent publication'). Placeholder."""
        # This requires more sophisticated date handling and comparison logic
        # Example: Flag publications within the last 2 years as 'recent'
        # annotations = {}
        # if 'publications' in structured_data and isinstance(structured_data['publications'], list):
        #     recent_pubs = []
        #     current_year = datetime.now().year
        #     for pub in structured_data['publications']:
        #         pub_year = pub.get('year') # Assuming 'year' field exists
        #         if pub_year and isinstance(pub_year, int) and current_year - pub_year <= 2:
        #              recent_pubs.append(pub.get('title', 'Unknown Title'))
        #     if recent_pubs:
        #          annotations['recent_publications'] = recent_pubs
        # logger.info("Temporal annotation check complete.")
        # return annotations
        logger.info("Temporal annotation check skipped (placeholder).")
        return {} # Placeholder


    def run_quality_checks(self, structured_data: Dict, schema: Dict) -> Dict:
        """Orchestrates all quality checks."""
        logger.info("Running quality control checks...")
        completeness_score, missing_required = self._check_completeness(structured_data, schema)
        duplicates_found = self._detect_duplicates(structured_data, schema)
        temporal_annotations = self._annotate_temporal_relevance(structured_data, schema) # Placeholder

        qc_report = {
            "completeness_score": completeness_score,
            "missing_required_fields": missing_required,
            "potential_duplicates": duplicates_found,
            "temporal_annotations": temporal_annotations, # Placeholder
            "notes": []
        }

        if completeness_score < 0.7:
             qc_report["notes"].append("Low data completeness.")
        if duplicates_found:
             qc_report["notes"].append("Potential duplicate items detected in lists.")

        logger.info("Quality control checks completed.")
        return qc_report
