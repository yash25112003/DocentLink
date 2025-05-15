import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from google.adk.agents import Agent # Assuming this is the correct import path
from google.adk.models.lite_llm import LiteLlm # Assuming this is the correct import path
from google.adk.sessions import InMemorySessionService # Assuming this is the correct import path
from google.adk.runners import Runner # Assuming this is the correct import path
from google.adk.tools.base_tool import BaseTool # Assuming this is the correct import path
from google.adk.tools.tool_context import ToolContext # Assuming this is the correct import path
# from google.genai import types # This import was present but not used, can be removed if not needed elsewhere
from pymongo import MongoClient
import re
import asyncio
from google.genai.types import UserContent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model constants
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash" # Make sure this model name is valid for LiteLlm

__all__ = ['EmailAgentSystem']

class UserDataTool(BaseTool):
    def __init__(self, fetch_function):
        super().__init__(
            name="fetch_user_data",
            description="Fetch user data from the database given a user_id. Input must be a dictionary with 'user_id'."
        )
        self._fetch_function = fetch_function

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        user_id = args.get('user_id')
        if not user_id:
            return {"error": "user_id is required for fetch_user_data"}
        return await self._fetch_function(user_id)

class ProfessorDataTool(BaseTool):
    def __init__(self, fetch_function):
        super().__init__(
            name="fetch_professor_data",
            description="Fetch professor data from the database given a professor_name. Input must be a dictionary with 'professor_name'."
        )
        self._fetch_function = fetch_function

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        professor_name = args.get('professor_name')
        if not professor_name:
            return {"error": "professor_name is required for fetch_professor_data"}
        return await self._fetch_function(professor_name)

class EmailGeneratorTool(BaseTool):
    def __init__(self, generate_function):
        super().__init__(
            name="generate_email",
            description="Generate a personalized email using user_data and professor_data. Input must be a dictionary with 'user_data' and 'professor_data'."
        )
        self._generate_function = generate_function

    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        user_data = args.get('user_data')
        professor_data = args.get('professor_data')
        if not user_data or not professor_data:
            return {"error": "user_data and professor_data are required for generate_email"}
        return await self._generate_function(user_data, professor_data)

class EmailAgentSystem:
    def __init__(self, session_state: Dict, session_service=None):
        """Initialize the email agent system with session state."""
        self.session_state = session_state
        self.db_client = None
        # Ensure MONGODB_URI is set as an environment variable
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            logger.error("MONGODB_URI environment variable not set.")
            raise ValueError("MONGODB_URI environment variable not set.")
        self.db_client = MongoClient(mongodb_uri)

        self.model = LiteLlm(model=MODEL_GEMINI_2_0_FLASH)
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
            model=self.model,
            description="Database operations specialist that handles user and professor data retrieval.",
            instruction="You are a database assistant. You fetch user and professor data when requested.",
            tools=[self.user_data_tool, self.professor_data_tool],
            before_model_callback=lambda **kwargs: self._before_model_callback(**kwargs),
            before_tool_callback=self._before_tool_callback
        )

        self.professor_agent = Agent(
            name="professor_agent",
            model=self.model,
            description="Professor data specialist that handles professor profile information.",
            instruction="You are a professor data assistant. You fetch professor data when requested.",
            tools=[self.professor_data_tool],
            before_model_callback=lambda **kwargs: self._before_model_callback(**kwargs),
            before_tool_callback=self._before_tool_callback
        )

        self.llm_agent = Agent(
            name="llm_agent",
            model=self.model,
            description="Email generation specialist that creates personalized academic emails.",
            instruction="You are an email generation assistant. You create personalized emails based on user and professor data.",
            tools=[self.email_generator_tool],
            before_model_callback=lambda **kwargs: self._before_model_callback(**kwargs),
            before_tool_callback=self._before_tool_callback
        )

        # Then initialize the root agent that coordinates the sub-agents
        root_instruction = (
            "You are the main coordinator for generating personalized academic emails. "
            "Your primary responsibility is to orchestrate the email generation process. "
            "You have specialized sub-agents: "
            "1. 'database_agent': Handles fetching user and professor data. "
            "2. 'professor_agent': Handles professor profile information. "
            "3. 'llm_agent': Handles the actual email generation. "
            "Follow these steps: "
            "1. Use the 'fetch_user_data' tool to get the user's profile using the provided user_id. "
            "2. Use the 'fetch_professor_data' tool to get the professor's profile using the professor_name. "
            "3. Use the 'generate_email' tool with the fetched user_data and professor_data to create the email. "
            "4. Return the generated email content (subject and body). "
            "Ensure you pass the correct arguments to each tool as per their descriptions."
        )

        self.root_agent = Agent(
            name="root_agent",
            model=self.model,
            description="Main coordinator for the email generation system.",
            instruction=root_instruction,
            tools=self.tools,  # Root agent needs access to all tools for orchestration
            sub_agents=[self.database_agent, self.professor_agent, self.llm_agent],
            before_model_callback=lambda **kwargs: self._before_model_callback(**kwargs),
            before_tool_callback=self._before_tool_callback
        )

    def _before_model_callback(self, **kwargs) -> Optional[str]:
        agent = kwargs.get("agent")
        prompt = kwargs.get("prompt")
        # Optionally: callback_context = kwargs.get("callback_context")
        # Optionally: llm_request = kwargs.get("llm_request")
        if agent is None or prompt is None:
            logger.error("_before_model_callback called without required 'agent' or 'prompt'. Got: %s", kwargs)
            return None
        sensitive_patterns = [
            r'\b\d{16}\b',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        ]
        
        if "generate_email" not in prompt.lower() and "user profile" not in prompt.lower():
            for pattern in sensitive_patterns:
                if re.search(pattern, prompt):
                    logger.warning(f"Sensitive information detected in prompt for {agent.name}")
        
        strict_inappropriate_keywords = ["password", "secret_key_do_not_reveal"]
        if any(keyword in prompt.lower() for keyword in strict_inappropriate_keywords):
            logger.warning(f"Highly inappropriate content detected in prompt for {agent.name}")
            return "Error: Inappropriate content detected in the prompt."

        return None

    async def _before_tool_callback(self, agent: Agent, tool_name: str, tool_args: Dict) -> Optional[Dict]:
        """Safety guardrail for tool arguments."""
        if tool_name == "fetch_user_data":
            if not tool_args.get("user_id"):
                logger.error(f"Missing user_id for {tool_name} by {agent.name}")
                return {"error": "Missing required user_id parameter for fetch_user_data"}
            
        elif tool_name == "fetch_professor_data":
            if not tool_args.get("professor_name"):
                logger.error(f"Missing professor_name for {tool_name} by {agent.name}")
                return {"error": "Missing required professor_name parameter for fetch_professor_data"}
            
        elif tool_name == "generate_email":
            required_fields = ["user_data", "professor_data"]
            missing_fields = [field for field in required_fields if not tool_args.get(field)]
            if missing_fields:
                logger.error(f"Missing fields for {tool_name} by {agent.name}: {missing_fields}")
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
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(base_dir, '..', 'scrapper', 'output')
            
            if not os.path.isdir(output_dir):
                logger.error(f"Professor data directory not found: {output_dir}")
                return {"error": f"Professor data directory not found: {output_dir}"}

            professor_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
            found_data = {}
            
            normalized_prof_name = professor_name.lower().replace(" ", "_")

            for file_name in professor_files:
                normalized_file_name_part = file_name.replace('.json', '').lower()
                if normalized_prof_name in normalized_file_name_part or \
                   professor_name.lower() in file_name.lower():
                    file_path = os.path.join(output_dir, file_name)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            found_data.update(data) 
                            logger.info(f"Loaded data for {professor_name} from {file_name}")
                    except json.JSONDecodeError as je:
                        logger.error(f"JSON decode error in file {file_path}: {str(je)}")
                    except Exception as fe:
                        logger.error(f"Error reading or processing file {file_path}: {str(fe)}")
            
            logger.debug(f"_fetch_professor_data for {professor_name}: Found type {type(found_data)}")
            if not found_data:
                logger.warning(f"No data found for professor: {professor_name} in {output_dir}")
                return {"error": f"No data found for professor: {professor_name}"}
            return found_data
        except Exception as e:
            logger.error(f"Error fetching professor data for {professor_name}: {str(e)}", exc_info=True)
            return {"error": f"Error fetching professor data: {str(e)}"}

    async def _generate_email(self, user_data: Dict, professor_data: Dict) -> Dict:
        """Generate personalized email using Gemini."""
        try:
            if not isinstance(user_data, dict) or not isinstance(professor_data, dict):
                err_msg = "Invalid input: user_data and professor_data must be dictionaries."
                logger.error(f"{err_msg} Got user_data: {type(user_data)}, professor_data: {type(professor_data)}")
                return {"error": err_msg}
            if "error" in user_data:
                return {"error": f"Cannot generate email due to user data error: {user_data['error']}"}
            if "error" in professor_data:
                return {"error": f"Cannot generate email due to professor data error: {professor_data['error']}"}

            logger.debug(f"_generate_email: user_data keys: {user_data.keys()}, professor_data keys: {professor_data.keys()}")

            user_name = user_data.get('name', 'the student')
            user_email = user_data.get('email', '')
            user_resume_analysis = user_data.get('resume_analysis', {})
            user_phone = user_resume_analysis.get('contact', {}).get('phone', '[Your Phone]') if isinstance(user_resume_analysis.get('contact'),dict) else '[Your Phone]'
            user_university = user_data.get('university', '[Your University]')
            user_location = user_resume_analysis.get('contact', {}).get('location', '[Your Location]') if isinstance(user_resume_analysis.get('contact'),dict) else '[Your Location]'
            user_resume_link = user_data.get('resume_link', '[Link to Your Resume]')

            prof_name = professor_data.get('name', 'Professor')

            prompt = f"""
            You are an expert academic email writer. Generate a professional and personalized academic email.

            User Profile:
            Name: {user_name}
            Email: {user_email}
            Phone: {user_phone}
            University: {user_university}
            Location: {user_location}
            Resume: {user_resume_link}
            Additional User Details: {json.dumps(user_data, indent=2, default=str)}

            Professor Profile:
            Professor's Name (if known from data): {prof_name}
            Professor's Details: {json.dumps(professor_data, indent=2, default=str)}

            Task: Write an email from the user to the professor expressing interest in their research and a potential research internship.

            Requirements:
            1. Professional academic tone.
            2. Word count: Approximately 300-350 words for the body.
            3. Specific references: If possible, identify 1-2 specific areas of the professor's work (from "Professor's Details") that align with the user's interests or background (from "User Profile" or "Additional User Details").
            4. Clear connection: Briefly explain how the user's background, skills, or interests make them a good fit for research in the professor's lab/area.
            5. Personalized Subject Line: Create a subject line like "Research Internship Inquiry - [User's Key Area of Interest] - {user_name}".
            6. Email Structure:

            Subject: [Generated Personalized Subject Line]

            Body:
            Dear Professor {prof_name if prof_name != 'Professor' else '[Professor Last Name]'},

            My name is {user_name}, and I am a [e.g., final-year Computer Engineering student] at {user_university}. I am writing to express my keen interest in your research, particularly in [mention specific area from professor's profile].

            [Paragraph 2: Elaborate on your interest. Refer to a specific paper, project, or research theme of the professor. Connect it to your own studies, projects, or skills. For example: "I was particularly fascinated by your recent work on [specific project/paper title] because [reason related to your skills/interests from user_data like resume_analysis or projects]. My experience in [relevant skill/project from user_data] has prepared me to contribute effectively to such research."]

            [Paragraph 3: Briefly state your goal, e.g., seeking a research internship for Summer 202X or during the upcoming academic year. Mention your availability if known.] I am highly motivated to contribute to cutting-edge research at your esteemed institution and gain further hands-on experience.

            I have attached my resume for your review at: {user_resume_link}. I would be grateful for the opportunity to discuss how my background and enthusiasm could benefit your research endeavors.

            Thank you for your time and consideration.

            Sincerely,
            {user_name}
            {user_university}
            {user_location if user_location != '[Your Location]' else ''}
            {user_phone if user_phone != '[Your Phone]' else ''}
            {user_email}

            Output Format:
            Return the email as a JSON object with two keys: "subject" and "body".
            Ensure the body is between 300 and 350 words.
            """

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.model.generate_content, prompt)
            
            if not response or not hasattr(response, 'text') or not response.text:
                raise ValueError("Failed to generate email content or received empty response from LLM.")
            
            try:
                email_data_str = response.text
                if email_data_str.strip().startswith("```json"):
                    email_data_str = email_data_str.strip()[7:-3].strip()
                elif email_data_str.strip().startswith("```"):
                     email_data_str = email_data_str.strip()[3:-3].strip()

                email_data = json.loads(email_data_str)
            except json.JSONDecodeError as je:
                logger.error(f"LLM response is not valid JSON. Error: {je}. Response text: {response.text}")
                if "Subject:" in response.text and "Body:" in response.text:
                    logger.warning("Attempting to parse Subject/Body from non-JSON LLM response.")
                    subject_match = re.search(r"Subject:([^\n]+)", response.text, re.IGNORECASE)
                    body_match = re.search(r"Body:(.*)", response.text, re.DOTALL | re.IGNORECASE)
                    if subject_match and body_match:
                        email_data = {"subject": subject_match.group(1).strip(), "body": body_match.group(1).strip()}
                    else:
                        raise ValueError(f"LLM did not return valid JSON and fallback parsing failed. Response: {response.text}")
                else:
                    raise ValueError(f"LLM did not return valid JSON. Response: {response.text}")


            if not isinstance(email_data, dict) or 'subject' not in email_data or 'body' not in email_data:
                logger.error(f"LLM response missing 'subject' or 'body': {email_data}")
                raise ValueError("LLM response missing 'subject' or 'body' fields.")

            word_count = len(email_data['body'].split())
            if not (280 <= word_count <= 370):
                logger.warning(f"Email body word count ({word_count}) outside preferred range (300-350). LLM may need prompt refinement if this is critical.")

            return email_data
        except Exception as e:
            logger.error(f"Error generating email: {str(e)}", exc_info=True)
            return {"error": f"Exception in email generation: {str(e)}"}

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
                    sanitized_professor_name = re.sub(r'\W+', '_', professor_name.lower())
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                    session_id = f"session_{user_id}_{sanitized_professor_name}_{timestamp}"
                    
                    session_state = {
                        "user_id": user_id,
                        "professor_name": professor_name
                    }
                    self.session_service.create_session(
                        app_name="ask_my_prof_email_generation",
                        user_id=user_id,
                        session_id=session_id,
                        state=session_state
                    )
                    runner = Runner(
                        agent=self.root_agent,
                        app_name="ask_my_prof_email_generation",
                        session_service=self.session_service
                    )
                    
                    final_agent_response = None
                    prompt_message = (
                        f"Generate a personalized email to Professor '{professor_name}' "
                        f"from the user associated with user_id '{user_id}'. "
                        "Use available tools to fetch user details and professor details first, then generate the email."
                    )
                    new_message = UserContent(parts=[{"text": prompt_message}])
                    
                    logger.info(f"Running agent for user '{user_id}', professor '{professor_name}', session '{session_id}'")
                    
                    # Handle the generator returned by runner.run()
                    agent_responses = runner.run(
                        session_id=session_id,
                        new_message=new_message,
                        user_id=user_id
                    )
                    
                    # Process all responses from the generator
                    for agent_response in agent_responses:
                        logger.debug(f"Agent response part for {professor_name}: {type(agent_response)}")
                        final_agent_response = agent_response

                    if final_agent_response and hasattr(final_agent_response, 'output'):
                        email_content = final_agent_response.output
                        if isinstance(email_content, dict) and 'subject' in email_content and 'body' in email_content:
                            results[professor_name] = {
                                'status': 'success',
                                'email': email_content
                            }
                            logger.info(f"Successfully generated email for {professor_name}")
                        elif isinstance(email_content, dict) and 'error' in email_content:
                            logger.error(f"Agent returned an error for {professor_name}: {email_content['error']}")
                            results[professor_name] = {
                                'status': 'error',
                                'error': f"Agent error: {email_content['error']}"
                            }
                        else:
                            logger.error(f"Invalid or unexpected email content format from agent for {professor_name}: {email_content}")
                            results[professor_name] = {
                                'status': 'error',
                                'error': 'Invalid email content format from agent. Output was: ' + str(email_content)[:200]
                            }
                    else:
                        logger.error(f"No valid final response from agent for {professor_name}. Last part: {final_agent_response}")
                        results[professor_name] = {
                            'status': 'error',
                            'error': 'No conclusive response from email generation agent'
                        }
                except Exception as e:
                    logger.error(f"Unhandled exception generating email for {professor_name}: {str(e)}", exc_info=True)
                    results[professor_name] = {
                        'status': 'error',
                        'error': f"Outer exception: {str(e)}"
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

    base_dir = os.path.dirname(os.path.abspath(__file__))
    scrapper_output_dir = os.path.join(base_dir, '..', 'scrapper', 'output')
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
        with open(os.path.join(scrapper_output_dir, 'dr_ada_lovelace.json'), 'w') as f:
            json.dump(prof1_data, f, indent=2)
        with open(os.path.join(scrapper_output_dir, 'professor_charles_babbage.json'), 'w') as f:
            json.dump(prof2_data, f, indent=2)
        logger.info("Created dummy professor JSON files.")
    except Exception as e:
        logger.error(f"Could not create dummy professor files: {e}")

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

    except ValueError as ve:
        logger.error(f"Initialization error: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during main_test: {e}", exc_info=True)
    finally:
        if email_system:
            email_system.close_connections()

if __name__ == "__main__":
    try:
        asyncio.run(main_test())
    except RuntimeError as re_err:
        if "cannot run current event loop" in str(re_err) and "nest_asyncio" not in str(re_err):
            print("RuntimeError with event loop. If you are in an environment like Jupyter, "
                  "try installing 'nest_asyncio' and adding 'import nest_asyncio; nest_asyncio.apply()' at the top.")
        raise