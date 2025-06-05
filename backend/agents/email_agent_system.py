import os
import json
import logging
import time
import re
import asyncio
import google.generativeai as genai
from typing import Dict, List, Optional, Any
from datetime import datetime
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from pymongo import MongoClient
from google.genai.types import UserContent, GenerationConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
try:
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    logger.info("Gemini API configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}", exc_info=True)

# Model constants
MODEL_NAME = "gemini-1.5-flash"
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 1
LLM_TEMPERATURE = 0.7

def _call_gemini_api(prompt: str, retries: int = LLM_MAX_RETRIES, delay: int = LLM_RETRY_DELAY) -> Optional[str]:
    """Calls the Gemini API with retry logic."""
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Cannot call Gemini API: API key not configured.")
        return None

    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            # Pass generation config as a dictionary instead of GenerationConfig object
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": LLM_TEMPERATURE,
                    "candidate_count": 1,
                    "max_output_tokens": 2048,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            if response and hasattr(response, 'text'):
                return response.text.strip()
            else:
                logger.warning(f"Gemini API returned unexpected response structure on attempt {attempt + 1}")
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

class UserDataTool(BaseTool):
    def __init__(self, fetch_function):
        super().__init__(
            name="fetch_user_data",
            description="Fetch user data from the database. Requires 'user_id' parameter."
        )
        self._fetch_function = fetch_function

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        user_id = args.get('user_id')
        if not user_id:
            return {"error": "user_id is required"}
        return await self._fetch_function(user_id)

class ProfessorDataTool(BaseTool):
    def __init__(self, fetch_function):
        super().__init__(
            name="fetch_professor_data",
            description="Fetch professor data from files. Requires 'professor_name' parameter."
        )
        self._fetch_function = fetch_function

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        professor_name = args.get('professor_name')
        if not professor_name:
            return {"error": "professor_name is required"}
        return await self._fetch_function(professor_name)

class EmailGeneratorTool(BaseTool):
    def __init__(self, generate_function):
        super().__init__(
            name="generate_email",
            description="Generate an email using provided user and professor data. Requires 'user_data' and 'professor_data' parameters."
        )
        self._generate_function = generate_function

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        user_data = args.get('user_data')
        professor_data = args.get('professor_data')
        if not user_data or not professor_data:
            return {"error": "Both user_data and professor_data are required"}
        return await self._generate_function(user_data, professor_data)

class EmailAgentSystem:
    def __init__(self, session_state: Dict, session_service=None):
        """Initialize the email agent system."""
        self.session_state = session_state
        self.db_client = None
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            logger.error("MONGODB_URI environment variable not set.")
            raise ValueError("MONGODB_URI environment variable not set.")
        self.db_client = MongoClient(mongodb_uri)

        self.session_service = session_service or InMemorySessionService()
        self._setup_tools()
        self._initialize_agents()

    def _setup_tools(self):
        """Set up the tools for the agents."""
        self.user_data_tool = UserDataTool(self._fetch_user_data)
        self.professor_data_tool = ProfessorDataTool(self._fetch_professor_data)
        self.email_generator_tool = EmailGeneratorTool(self._generate_email)
        
        self.tools = [
            self.user_data_tool,
            self.professor_data_tool,
            self.email_generator_tool
        ]

    def _initialize_agents(self):
        """Initialize the agent team."""
        # First initialize the specialized agents
        self.database_agent = Agent(
            name="database_agent",
            model=MODEL_NAME,
            description="Database operations specialist that handles user and professor data retrieval.",
            instruction="""You are a database assistant. Your primary role is to fetch data using the tools provided.
            When you receive a request:
            1. If it mentions a user_id, use fetch_user_data with that id
            2. If it mentions a professor name, use fetch_professor_data with that name
            3. If you get both pieces of data, pass them to generate_email
            4. Only transfer to another agent if you've tried using your tools first
            """,
            tools=[self.user_data_tool, self.professor_data_tool],
            before_model_callback=self._before_model_callback,
            before_tool_callback=self._before_tool_callback
        )

        self.professor_agent = Agent(
            name="professor_agent",
            model=MODEL_NAME,
            description="Professor data specialist that handles professor profile information.",
            instruction="""You are a professor data assistant. Your primary role is to process professor information.
            When you receive a request:
            1. Use fetch_professor_data with the professor's name to get their information
            2. Only transfer to another agent if you've tried using your tools first
            """,
            tools=[self.professor_data_tool],
            before_model_callback=self._before_model_callback,
            before_tool_callback=self._before_tool_callback
        )

        self.llm_agent = Agent(
            name="llm_agent",
            model=MODEL_NAME,
            description="Email generation specialist that creates personalized academic emails.",
            instruction="""You are an email generation assistant. Your primary role is to create personalized emails.
            When you receive a request:
            1. Check if you have both user_data and professor_data
            2. If you have both, use generate_email to create the email
            3. If you're missing data, transfer to database_agent to fetch it
            """,
            tools=[self.email_generator_tool],
            before_model_callback=self._before_model_callback,
            before_tool_callback=self._before_tool_callback
        )

        # Root agent coordinates the process
        root_instruction = """You are the main coordinator for generating personalized academic emails.
        When generating an email, follow these steps in order:
        1. Use fetch_user_data with the user_id to get their profile
        2. Use fetch_professor_data with the professor name to get their information
        3. Once you have both pieces of data, use generate_email to create the email
        4. Return the generated email content with subject and body

        Do not transfer to other agents unless you've tried using the appropriate tools first.
        """

        self.root_agent = Agent(
            name="root_agent",
            model=MODEL_NAME,
            description="Main coordinator for the email generation system.",
            instruction=root_instruction,
            tools=self.tools,
            sub_agents=[self.database_agent, self.professor_agent, self.llm_agent],
            before_model_callback=self._before_model_callback,
            before_tool_callback=self._before_tool_callback
        )

    # Updated signature to accept explicitly passed arguments from the lambda
    def _before_model_callback(self, callback_context=None, llm_request=None, **kwargs) -> Optional[str]:
        """Pre-LLM call safety check."""
        # Safely extract agent from callback_context or kwargs
        agent = None
        if callback_context and hasattr(callback_context, 'agent'):
            agent = callback_context.agent
        elif 'agent' in kwargs:
            agent = kwargs['agent']

        # Even if agent is None, we should still perform basic checks on the request
        if llm_request is None:
            logger.error(f"_before_model_callback: 'llm_request' not found")
            return None

        # Extract text from LLM request contents
        prompt_text_parts = []
        if hasattr(llm_request, 'contents'):
            for content in llm_request.contents:
                if hasattr(content, 'parts'):
                    for part in content.parts:
                        if hasattr(part, 'text'):
                            prompt_text_parts.append(part.text)

        # Include system instruction if present
        if hasattr(llm_request, 'config') and hasattr(llm_request.config, 'system_instruction'):
            if isinstance(llm_request.config.system_instruction, str):
                prompt_text_parts.append(llm_request.config.system_instruction)
            elif hasattr(llm_request.config.system_instruction, 'text'):
                prompt_text_parts.append(llm_request.config.system_instruction.text)

        full_prompt = " ".join(prompt_text_parts).strip().lower()
        
        if not full_prompt:
            logger.warning(f"Empty prompt text for agent {agent.name if agent else 'unknown'}")
            return None

        # Skip PII checks for known email-related prompts
        if any(term in full_prompt for term in ["generate_email", "user profile", "fetch_user_data", "fetch_professor_data"]):
            return None

        # Check for sensitive information patterns
        sensitive_patterns = [
            r'\b\d{16}\b',  # Credit card numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Phone numbers
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, full_prompt):
                logger.warning(f"Sensitive pattern detected in prompt for {agent.name if agent else 'unknown'}")
                return "Error: Sensitive information detected in prompt"

        # Block strictly inappropriate content
        strict_inappropriate_keywords = ["password", "secret_key_do_not_reveal"]
        if any(keyword in full_prompt for keyword in strict_inappropriate_keywords):
            logger.error(f"Inappropriate content detected in prompt for {agent.name if agent else 'unknown'}")
            return "Error: Inappropriate content detected"

        return None

    # Updated signature to accept all ADK-provided arguments, including 'tool_obj' and 'callback_context_obj'
    async def _before_tool_callback(self, **kwargs) -> Optional[Dict]:
        """Safety guardrail for tool arguments."""
        # Extract all possible parameters from kwargs
        callback_context = kwargs.get('callback_context')
        agent = None
        tool_name = kwargs.get('tool_name')
        tool_args = kwargs.get('tool_args', {})
        
        # Try to get agent from different possible sources
        if callback_context and hasattr(callback_context, 'agent'):
            agent = callback_context.agent
        elif 'agent' in kwargs:
            agent = kwargs.get('agent')
        
        # Even without an agent, we can still validate the tool args
        logger.debug(f"Before tool callback: Tool='{tool_name}', Args='{tool_args}'")

        if tool_name == "fetch_user_data":
            if not tool_args.get("user_id"):
                logger.error(f"Missing user_id for {tool_name}")
                return {"error": "Missing required user_id parameter for fetch_user_data"}
            
        elif tool_name == "fetch_professor_data":
            if not tool_args.get("professor_name"):
                logger.error(f"Missing professor_name for {tool_name}")
                return {"error": "Missing required professor_name parameter for fetch_professor_data"}
            
        elif tool_name == "generate_email":
            required_fields = ["user_data", "professor_data"]
            missing_fields = [field for field in required_fields if not tool_args.get(field)]
            if missing_fields:
                logger.error(f"Missing fields for {tool_name}: {missing_fields}")
                return {"error": f"Missing required fields for generate_email: {', '.join(missing_fields)}"}

        return None

    async def _fetch_user_data(self, user_id: str) -> Dict:
        """Fetch user data from MongoDB."""
        try:
            db = self.db_client['ask_my_prof']
            user_data = db.user_profiles.find_one({'user_id': user_id}, {'_id': 0})
            logger.debug(f"_fetch_user_data for {user_id}: Found type {type(user_data)}")
            if not user_data:
                logger.warning(f"User not found: {user_id}")
                return {"error": f"User not found: {user_id}"}
            return user_data
        except Exception as e:
            logger.error(f"Error fetching user data for {user_id}: {str(e)}", exc_info=True)
            return {"error": f"Database error fetching user data: {str(e)}"}


    async def _fetch_professor_data(self, professor_name: str) -> Dict:
        """Fetch professor data from JSON files."""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, 'backend', 'scrapper', 'output')
            
            if not os.path.isdir(output_dir):
                logger.error(f"Professor data directory not found: {output_dir}")
                return {"error": f"Professor data directory not found: {output_dir}"}

            professor_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
            
            # Normalize the professor name for comparison
            normalized_prof_name = professor_name.lower().replace(" ", "_").replace(".", "")
            logger.info(f"Looking for professor {professor_name} (normalized: {normalized_prof_name}) in {output_dir}")
            logger.info(f"Available files: {professor_files}")

            # Try exact match first
            exact_match_file = None
            for file_name in professor_files:
                normalized_file_name = file_name.lower().replace(".json", "").replace(".", "")
                if normalized_prof_name == normalized_file_name or \
                   normalized_prof_name in normalized_file_name:
                    exact_match_file = file_name
                    break
            
            # If no exact match, try partial match
            if not exact_match_file:
                for file_name in professor_files:
                    normalized_file_name = file_name.lower().replace(".json", "").replace(".", "")
                    name_parts = normalized_prof_name.split("_")
                    if any(part in normalized_file_name for part in name_parts if len(part) > 2):
                        exact_match_file = file_name
                        break

            if exact_match_file:
                file_path = os.path.join(output_dir, exact_match_file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        logger.info(f"Loaded data for {professor_name} from {exact_match_file}")
                        return data
                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error in file {file_path}: {str(je)}")
                except Exception as fe:
                    logger.error(f"Error reading or processing file {file_path}: {str(fe)}")
            
            logger.warning(f"No matching data found for professor: {professor_name} in {output_dir}")
            return {"error": f"No data found for professor: {professor_name}"}

        except Exception as e:
            logger.error(f"Error fetching professor data for {professor_name}: {str(e)}", exc_info=True)
            return {"error": f"Error fetching professor data: {str(e)}"}

    async def _generate_email(self, user_data: dict, professor_data: dict) -> dict:
        """
        Generate personalized email using Gemini, prioritizing resume-extracted data
        and including resume contact details in the signature.
        """
        try:
            if not isinstance(user_data, dict) or not isinstance(professor_data, dict):
                err_msg = "Invalid input: user_data and professor_data must be dictionaries."
                logger.error(f"{err_msg} Got user_data: {type(user_data)}, professor_data: {type(professor_data)}") #
                return {"error": err_msg}
            if "error" in user_data:
                return {"error": f"Cannot generate email due to user data error: {user_data['error']}"} #
            if "error" in professor_data:
                return {"error": f"Cannot generate email due to professor data error: {professor_data['error']}"} #

            # --- Enhanced Data Extraction from resume_analysis ---
            resume_analysis = user_data.get('resume_analysis', {}) #
            # Ensure raw fields are fetched, default to empty strings if not present
            contact_details_raw = str(resume_analysis.get('contact_details_raw', ''))
            education_summary_raw = str(resume_analysis.get('education_summary_raw', resume_analysis.get('education', '')))

            # 1. User's Name (from resume analysis)
            user_name_from_resume = "the student"
            name_match_rag = re.search(r"candidate's name is ([A-Za-z\s\.]+)\.", contact_details_raw, re.IGNORECASE)
            if name_match_rag:
                user_name_from_resume = name_match_rag.group(1).strip()
            elif resume_analysis.get('candidate_name_from_resume') and resume_analysis['candidate_name_from_resume'] not in ["Unknown", "the student"]:
                 user_name_from_resume = resume_analysis['candidate_name_from_resume']
            logger.info(f"Using candidate name from resume: {user_name_from_resume}")

            # 2. User's University (from resume analysis)
            user_university_from_resume = "[User's University from Resume]"
            # More robust regex for university: looks for institution names before common delimiters
            uni_match_rag = re.search(
                r"^(.*?)(?:Degree:|Bachelor of|Diploma of|Expected Graduation:|CGPA:|GPA:|,?\s*(?:Mumbai|India|Maharashtra|Arizona|Stanford|Cambridge|MIT|Harvard|Berkeley|USA|UK))",
                education_summary_raw, re.IGNORECASE | re.DOTALL
            )
            if uni_match_rag:
                parsed_uni = uni_match_rag.group(1).strip()
                # Clean up common RAG prefixes
                if parsed_uni.lower().startswith("the candidate has the following education:"):
                    parsed_uni = parsed_uni[len("the candidate has the following education:"):].strip()
                if parsed_uni.lower().startswith("summarize the candidate's education"): # Another possible prefix
                    parsed_uni = parsed_uni[len("summarize the candidate's education"):].strip()
                if parsed_uni and len(parsed_uni) > 5 and not parsed_uni.lower().startswith("answer:"): # Basic sanity check
                    user_university_from_resume = parsed_uni
            elif resume_analysis.get('university_from_resume') and resume_analysis['university_from_resume'] not in ["Unknown", "[User's University from Resume]"]:
                user_university_from_resume = resume_analysis['university_from_resume']
            logger.info(f"Using candidate university from resume: {user_university_from_resume}")

            # Extract user's email
            user_email_from_resume = None
            email_match_rag = re.search(r"Email:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", contact_details_raw, re.IGNORECASE)
            if email_match_rag:
                user_email_from_resume = email_match_rag.group(1).strip()
            elif resume_analysis.get('email_from_resume'):
                user_email_from_resume = resume_analysis['email_from_resume']
            logger.info(f"Using candidate email from resume: {user_email_from_resume}")

            # Extract user's phone with more robust pattern matching
            user_phone_from_resume = None
            phone_match_rag = re.search(r"Phone:\s*(\+?[0-9\s\-\(\)]{7,})", contact_details_raw, re.IGNORECASE)
            if phone_match_rag:
                user_phone_from_resume = phone_match_rag.group(1).strip()
            elif resume_analysis.get('phone_from_resume'):
                user_phone_from_resume = resume_analysis['phone_from_resume']
            logger.info(f"Using candidate phone from resume: {user_phone_from_resume}")

            # Prepare signature block with consistent formatting
            signature_lines = ["Sincerely,", "", user_name_from_resume]
            if user_email_from_resume:
                signature_lines.append(f"Email: {user_email_from_resume}")  # Add "Email:" prefix
            if user_phone_from_resume:
                signature_lines.append(f"Phone: {user_phone_from_resume}")  # Add "Phone:" prefix
            signature_block = "\n".join(signature_lines)

            # Generate email prompt with explicit signature instructions
            prompt = f"""
            You are an expert academic email writer. Generate a professional and personalized academic email.

            User Profile Information:
            Name: {user_name_from_resume}
            University: {user_university_from_resume or '[University not found in resume]'}
            Email: {user_email_from_resume}
            Phone: {user_phone_from_resume}
            Full User Profile:
            {json.dumps(user_data, indent=2, default=str)}

            Professor Profile:
            {json.dumps(professor_data, indent=2, default=str)}

            Task: Write an email expressing interest in research and potential internship opportunities.

            Critical Instructions:
            1. Use EXACTLY this signature block at the end (do not modify it):
            {signature_block}
            
            2. Keep the email professional, focused, and approximately 300-350 words.
            3. Reference specific areas of the professor's research that align with the student's background.
            4. DO NOT add any additional contact information beyond what is in the signature block.

            Output Format (Return ONLY valid JSON):
            {{
                "subject": "Research Internship Inquiry - [Research Area] - {user_name_from_resume}",
                "body": "The email content here... [must end with the exact signature block provided]"
            }}
            """

            logger.info("Calling Gemini API to generate email...")
            model = genai.GenerativeModel(MODEL_NAME)
            
            # Configure generation parameters directly in generate_content
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": LLM_TEMPERATURE,
                    "candidate_count": 1,
                    "max_output_tokens": 2048,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )

            if not response or not hasattr(response, 'text'):
                raise ValueError("Failed to generate email content or received empty response from LLM.")

            response_text = response.text.strip() #
            logger.debug(f"Raw LLM response: {response_text[:300]}...")

            try:
                # Clean the response if it contains markdown code fences
                if response_text.startswith("```json"):
                    response_text = response_text[7:-3].strip() #
                elif response_text.startswith("```"):
                    response_text = response_text[3:-3].strip() #

                email_data = json.loads(response_text) #
            except json.JSONDecodeError as je:
                logger.error(f"LLM response is not valid JSON. Error: {je}. Response text: {response_text}") #
                if "Subject:" in response_text and "Body:" in response_text:
                    logger.warning("Attempting to parse Subject/Body from non-JSON LLM response as a fallback.") #
                    subject_match = re.search(r"Subject:\s*(.+)", response_text, re.IGNORECASE)
                    # Ensure Body captures multi-line content and stops before any potential post-body text if LLM adds extra.
                    body_match = re.search(r"Body:\s*((?:.|\n)+)", response_text, re.IGNORECASE)
                    if subject_match and body_match:
                        email_data = {"subject": subject_match.group(1).strip(), "body": body_match.group(1).strip()}
                    else:
                        raise ValueError(f"LLM did not return valid JSON and fallback parsing failed. Response: {response_text}") #
                else:
                    raise ValueError(f"LLM did not return valid JSON. Response: {response_text}") #

            if not isinstance(email_data, dict) or 'subject' not in email_data or 'body' not in email_data:
                logger.error(f"LLM response missing 'subject' or 'body': {email_data}") #
                raise ValueError("LLM response missing 'subject' or 'body' fields after parsing.") #

            word_count = len(email_data['body'].split()) #
            logger.info(f"Generated email - Subject: {email_data['subject']}") #
            logger.info(f"Email body word count: {word_count}") #
            if not (250 <= word_count <= 450): # Relaxed word count
                logger.warning(f"Email body word count ({word_count}) outside typical range (300-350).") #

            return email_data #
        except Exception as e:
            logger.error(f"Error generating email: {str(e)}", exc_info=True) #
            return {"error": f"Exception in email generation: {str(e)}"} #
        
    async def generate_personalized_email(self) -> Dict:
        """Generate personalized emails for selected professors."""
        try:
            user_id = self.session_state.get('user_id')
            if not user_id:
                logger.error("User ID not found in session state for email generation.")
                return {
                    'status': 'error',
                    'error': "User ID not found in session state"
                }
            
            selected_professors = self.session_state.get('selected_professors', [])
            if not selected_professors:
                logger.warning("No professors selected for email generation.")
                return {
                    'status': 'error',
                    'error': "No professors selected"
                }

            results = {}
            for professor_name in selected_professors:
                try:
                    # First, fetch the required data directly
                    user_data = await self._fetch_user_data(user_id)
                    professor_data = await self._fetch_professor_data(professor_name)

                    # Check if we got valid data
                    if "error" in user_data:
                        results[professor_name] = {
                            'status': 'error',
                            'error': f"Failed to fetch user data: {user_data['error']}"
                        }
                        continue

                    if "error" in professor_data:
                        results[professor_name] = {
                            'status': 'error',
                            'error': f"Failed to fetch professor data: {professor_data['error']}"
                        }
                        continue

                    # Now generate the email with the collected data
                    email_result = await self._generate_email(user_data, professor_data)
                    
                    if "error" in email_result:
                        results[professor_name] = {
                            'status': 'error',
                            'error': f"Failed to generate email: {email_result['error']}"
                        }
                    else:
                        results[professor_name] = {
                            'status': 'success',
                            'email': email_result
                        }
                        logger.info(f"Successfully generated email for {professor_name}")

                except Exception as e:
                    logger.error(f"Unhandled exception generating email for {professor_name}: {str(e)}", exc_info=True)
                    results[professor_name] = {
                        'status': 'error',
                        'error': f"Exception: {str(e)}"
                    }

            return {
                'status': 'success',
                'emails': results
            }

        except Exception as e:
            logger.error(f"Critical error in generate_personalized_email orchestration: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': f"System error: {str(e)}"
            }
        finally:
            if self.db_client:
                logger.info("Closing MongoDB client in EmailAgentSystem.")
                self.db_client.close()
                self.db_client = None

    def close_connections(self):
        """Close any open connections."""
        if self.db_client:
            logger.info("Explicitly closing MongoDB client.")
            self.db_client.close()
            self.db_client = None


async def main_test():
    mock_session_state = {
        'user_id': 'test_user_123',
        'selected_professors': ["Dr. Ada Lovelace", "Professor Charles Babbage"],
    }

    if os.getenv("MONGODB_URI"):
        try:
            client = MongoClient(os.getenv("MONGODB_URI"))
            db = client['ask_my_prof']
            if not db.user_profiles.find_one({'user_id': 'test_user_123'}):
                db.user_profiles.insert_one({
                    'user_id': 'test_user_123',
                    'name': 'Test User',
                    'email': 'test.user@example.com',
                    'university': 'Test University',
                    'resume_analysis': {
                        'contact': {'phone': '123-456-7890', 'location': 'Test City, TS'},
                        'skills': ['Python', 'AI', 'Machine Learning'],
                        'projects': [{'title': 'AI Chatbot', 'description': 'Developed a chatbot.'}]
                    },
                    'resume_link': 'https://example.com/resume_test_user.pdf'
                })
                logger.info("Inserted dummy user profile for test_user_123.")
            client.close()
        except Exception as e:
            logger.error(f"Could not set up dummy user data: {e}")
    else:
        logger.warning("MONGODB_URI not set. Cannot create dummy user data.")

    # Fix path to match structure expected by _fetch_professor_data
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scrapper_output_dir = os.path.join(base_dir, 'backend', 'scrapper', 'output')
    os.makedirs(scrapper_output_dir, exist_ok=True)
    
    prof1_data = {
        "name": "Dr. Ada Lovelace",
        "university": "University of Analytics",
        "department": "Computer Science",
        "research_interests": ["Analytical Engines", "Computational Theory", "Symbolic AI"],
        "recent_publications": [
            {"title": "Notes on the Analytical Engine", "year": 1843},
            {"title": "Future of Computing Machines", "year": 1840}
        ],
        "email": "ada.lovelace@uanalytics.edu"
    }
    prof2_data = {
        "name": "Professor Charles Babbage",
        "university": "Cambridge Institute of Computation",
        "department": "Mechanical Engineering & Computation",
        "research_interests": ["Difference Engine", "Automated Computation", "Mechanical Calculators"],
        "contact": "cbabbage@cam.cic.edu"
    }
    
    try:
        prof1_file = os.path.join(scrapper_output_dir, 'dr_ada_lovelace.json')
        prof2_file = os.path.join(scrapper_output_dir, 'professor_charles_babbage.json')
        
        with open(prof1_file, 'w') as f:
            json.dump(prof1_data, f, indent=2)
        with open(prof2_file, 'w') as f:
            json.dump(prof2_data, f, indent=2)
        logger.info(f"Created test professor data files in {scrapper_output_dir}")
        logger.info(f"Files created: {os.listdir(scrapper_output_dir)}")
    except Exception as e:
        logger.error(f"Could not create professor data files: {e}", exc_info=True)

    if not os.getenv("MONGODB_URI"):
        logger.error("MONGODB_URI is not set. Aborting test.")
        return

    email_system = None
    try:
        email_system = EmailAgentSystem(session_state=mock_session_state)
        results = await email_system.generate_personalized_email()
        
        print("\n--- Email Generation Results ---")
        print(json.dumps(results, indent=2))

        if results.get('status') == 'success':
            for prof, data in results.get('emails', {}).items():
                if data.get('status') == 'success':
                    print(f"\n--- Email for {prof} ---")
                    print(f"Subject: {data['email']['subject']}")
                    print(f"Body:\n{data['email']['body']}")
                else:
                    print(f"\n--- Error for {prof} ---")
                    print(f"Error: {data.get('error')}")
        else:
            print(f"\nOverall Error: {results.get('error')}")

    except Exception as e:
        logger.error(f"Error in main_test: {e}", exc_info=True)
    finally:
        if email_system:
            email_system.close_connections()