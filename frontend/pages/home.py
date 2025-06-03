import streamlit as st
import os
import pandas as pd
from typing import List, Optional
import uuid
from datetime import datetime
import sys
import asyncio
import atexit
import subprocess  # Add this import

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from backend.agents.email_agent_system import EmailAgentSystem
from google.adk.sessions import InMemorySessionService
# from google.adk.runners import Runner # Runner is used within EmailAgentSystem, not directly here.

# Now you can import
from frontend.rag.resume_rag import resume_rag_main

# Add the backend directory to Python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend'))
sys.path.append(backend_path)

# Import the DatabaseManager
from db_manager import DatabaseManager # type: ignore

# Initialize database manager
db = DatabaseManager()

# Initialize session service for ADK (though EmailAgentSystem might create its own if not passed)
# This instance is used for the initial session creation if needed.
adk_session_service = InMemorySessionService()
APP_NAME = "ask_my_prof"
# USER_ID and SESSION_ID are specific to the ADK session, st.session_state.user_id is the primary user identifier.
# These global USER_ID and SESSION_ID for ADK might not be strictly necessary if EmailAgentSystem
# manages its own runner sessions based on st.session_state.user_id.
GLOBAL_ADK_USER_ID = None
GLOBAL_ADK_SESSION_ID = None


# Cache the professor data loading
@st.cache_data
def load_professor_data(university: str) -> List[str]:
    """
    Load professor data from university-specific CSV file.
    Returns list of professors for the given university.
    """
    try:
        # Sanitize university name for filename
        safe_university = sanitize_university_name(university)
        
        # Look for university-specific CSV file
        # Assuming this script is in frontend/pages/home.py
        # So ../../data/prof_data/ is correct if data is at project_root/data/prof_data
        csv_path = os.path.join(os.path.dirname(__file__), f'/Users/ShahYash/Desktop/Projects/ask-my-prof/AskMyProf/data/prof_data/{safe_university}.csv')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'Name' in df.columns:
                return df['Name'].dropna().unique().tolist() # Ensure unique and no NaN
            else:
                st.warning(f"CSV file for {university} exists at {csv_path} but doesn't have a 'Name' column.")
                return []
        else:
            st.warning(f"No CSV file found for {university} at {csv_path}")
            return []
            
    except Exception as e:
        st.error(f"Error loading professor data from {csv_path}: {str(e)}")
        return []

def sanitize_university_name(name: str) -> str:
    """Convert university name to lowercase with underscores for file naming"""
    return name.lower().replace(" ", "_").replace("&", "and") # Added handling for '&'

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'university' not in st.session_state:
    st.session_state.university = "Select University"
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'selected_professors' not in st.session_state:
    st.session_state.selected_professors = []
if 'uploaded_file_details' not in st.session_state: # To store name and avoid re-processing unnecessarily
    st.session_state.uploaded_file_details = None
if 'rag_results' not in st.session_state: # Ensure rag_results is initialized
    st.session_state.rag_results = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    user_data = {
        'user_id': st.session_state.user_id,
        'name': None,
        'email': None,
        'university': None,
        'professors': [],
        'file_name': None,
        'status': 'initialized', # Changed from 'incomplete'
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }
    db.create_user_profile(user_data)
    
    # Initialize ADK session (optional here, as EmailAgentSystem manages its own runner sessions)
    GLOBAL_ADK_USER_ID = st.session_state.user_id
    GLOBAL_ADK_SESSION_ID = f"session_streamlit_{GLOBAL_ADK_USER_ID}"
    # session = adk_session_service.create_session(
    #     app_name=APP_NAME,
    #     user_id=GLOBAL_ADK_USER_ID,
    #     session_id=GLOBAL_ADK_SESSION_ID,
    #     state={
    #         'user_id': GLOBAL_ADK_USER_ID, # Passed to EmailAgentSystem via st.session_state
    #         'status': 'initialized'
    #     }
    # )
    # st.session_state.adk_session_id = GLOBAL_ADK_SESSION_ID # Store if needed later

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #004d80;
    background: -webkit-linear-gradient(#004d80, #00b3b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}
.disabled {
    opacity: 0.5;
    pointer-events: none;
}
.stButton > button {
    width: 100%;
    margin-top: 1rem; /* Adjusted margin */
}
/* Ensure specific styling for buttons in columns if needed */
.button-row .stButton > button {
    margin-top: 0; /* Override general margin for buttons in a row */
}
</style>
""", unsafe_allow_html=True)

# Common title across pages
st.markdown("<div class='main-title'>AskMyProf</div>", unsafe_allow_html=True)

def update_user_data_in_db(update_fields: dict):
    """Helper function to update user data in MongoDB"""
    current_time = datetime.utcnow()
    update_fields['updated_at'] = current_time
    
    # Process RAG results for database update
    # ** CHANGE: Store under 'resume_analysis' key **
    # ** Also, try to structure more closely to what backend might expect, if possible from RAG **
    if 'rag_results' in st.session_state and st.session_state.rag_results:
        processed_resume_data = {}
        # Example: If RAG can provide structured contact info
        contact_info = {}
        
        for res in st.session_state.rag_results:
            query = res['query'].lower().strip()
            answer = res['answer']
            
            # This part needs careful alignment with resume_rag_main's output queries
            if 'name and contact details' in query:
                processed_resume_data['contact_details_raw'] = answer
                # Hypothetical: if RAG could parse phone/location
                # contact_info['phone'] = parsed_phone_from_answer
                # contact_info['location'] = parsed_location_from_answer
            elif 'technical skills' in query:
                processed_resume_data['skills'] = answer.split(',') if isinstance(answer, str) else answer # Store as list
            elif 'work experience' in query:
                processed_resume_data['work_experience'] = answer
            elif 'academic projects and papers' in query:
                processed_resume_data['projects_and_papers'] = answer
            elif 'courses and certifications' in query:
                processed_resume_data['courses_and_certifications'] = answer
            elif 'education' in query:
                processed_resume_data['education'] = answer
            elif 'programming languages' in query:
                processed_resume_data['programming_languages'] = answer.split(',') if isinstance(answer, str) else answer
        
        if contact_info: # If we had structured contact info
             processed_resume_data['contact'] = contact_info

        update_fields['resume_analysis'] = processed_resume_data # ** CHANGED KEY **
    
    db.update_user_profile(st.session_state.user_id, update_fields)
    
    # ADK session state update (if using a shared ADK session)
    # For now, EmailAgentSystem gets all it needs from st.session_state directly or via db tools.
    # if adk_session_service and GLOBAL_ADK_SESSION_ID:
    #     session = adk_session_service.get_session(app_name=APP_NAME, user_id=GLOBAL_ADK_USER_ID, session_id=GLOBAL_ADK_SESSION_ID)
    #     if session:
    #         session.state.update(update_fields)
    #         adk_session_service.update_session(session) # Persist the update

def page_one():
    """University Selection Page"""
    with st.form(key="page_one_form"):
        st.subheader("Step 1: University Selection")
        university_options = ["Select University", "Arizona State University", "Stanford University", "MIT", "Harvard University", "UC Berkeley"]
        current_uni_index = 0
        if st.session_state.university in university_options:
            current_uni_index = university_options.index(st.session_state.university)

        university = st.selectbox(
            "Select your university:",
            university_options,
            index=current_uni_index
        )

        submitted = st.form_submit_button("Continue to User Details")
        
        if submitted:
            if university == "Select University":
                st.error("Please select a university to continue.")
                return
            
            if st.session_state.university != university:
                st.session_state.selected_professors = [] # Reset if university changed
                
            st.session_state.university = university
            update_user_data_in_db({
                'university': university,
                'status': 'university_selected'
            })
            st.session_state.page = 2
            st.rerun()

def page_two():
    """User Details Page"""
    with st.form(key="page_two_form"):
        st.subheader("Step 2: User Details")
        user_email = st.text_input(
            "Your Email",
            value=st.session_state.user_email,
            placeholder="Enter your email address"
        )
        user_name = st.text_input(
            "Your Password",
            value=st.session_state.user_name,
            placeholder="Enter your Password",
            type="password"
        )

        col1, col2 = st.columns(2)
        with col1:
            back_button = st.form_submit_button("← Back to University Selection")
        with col2:
            continue_button = st.form_submit_button("Continue to Professor Selection →")

        if back_button:
            st.session_state.user_name = user_name # Save current input
            st.session_state.user_email = user_email
            st.session_state.page = 1
            st.rerun()
            
        if continue_button:
            if not user_name.strip():
                st.error("Please enter your name.")
                return
            if not user_email or "@" not in user_email or "." not in user_email: # Basic validation
                st.error("Please enter a valid email address.")
                return
            
            st.session_state.user_name = user_name
            st.session_state.user_email = user_email
            update_user_data_in_db({
                'name': user_name,
                'email': user_email,
                'status': 'user_details_completed'
            })
            st.session_state.page = 3
            st.rerun()

def page_three():
    """Professor Selection & File Upload Page"""
    st.subheader(f"Step 3: Professor & Resume for {st.session_state.university}")
    
    # Custom styles for this page (can be moved to main CSS block if preferred)
    st.markdown("""
    <style>
    .button-row { display: flex; justify-content: space-between; gap: 1rem; margin-top: 1.5rem; }
    .button-row .stButton > button { width: 100%; } /* Ensure buttons in row take full width of their column */
    .info-message { padding: 10px 15px; border-radius: 5px; margin-top: 0.5rem; background-color: #e8f0fe; border-left: 3px solid #4285f4; font-size: 0.9rem; }
    .success-message { background-color: #e6f4ea; color: #137333; padding: 15px; border-radius: 5px; margin-top: 1rem; text-align: center; font-weight: 500; border-left: 3px solid #137333; }
    .stMultiSelect > div > div { max-width: 100% !important; } /* Attempt to make multiselect wider */
    </style>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading professor data..."):
        professors_list = load_professor_data(st.session_state.university)
    
    if not professors_list:
        st.warning(f"No professors found for {st.session_state.university}. Please check the data source or select a different university.")
        if st.button("← Back to User Details"):
            st.session_state.page = 2
            st.rerun()
        return

    valid_selections = [p for p in st.session_state.selected_professors if p in professors_list]
    
    current_selection = st.multiselect(
        "Select up to 10 professors you are interested in:",
        options=professors_list,
        max_selections=10,
        default=valid_selections,
        placeholder="Type or select professor names"
    )
    
    st.markdown('<hr style="margin: 1rem 0; border-color: #e0e0e0;">', unsafe_allow_html=True)
    
    st.subheader("Upload Your Resume")
    uploaded_file_instance = st.file_uploader( # Use a different variable for the streamlit widget instance
        "Upload your resume (PDF, DOCX, TXT):",
        type=["pdf", "docx", "txt"],
        key="resume_file_uploader" # Unique key
    )
    
    if uploaded_file_instance is not None:
        # Check if it's a new file or the same one
        if st.session_state.uploaded_file_details is None or \
           st.session_state.uploaded_file_details['name'] != uploaded_file_instance.name or \
           st.session_state.uploaded_file_details['size'] != uploaded_file_instance.size:
            
            st.session_state.uploaded_file_details = {
                'name': uploaded_file_instance.name,
                'size': uploaded_file_instance.size,
                'type': uploaded_file_instance.type
            }
            st.session_state.rag_results = None # Reset previous RAG results for new file

            try:
                with st.spinner(f"Processing {uploaded_file_instance.name}..."):
                    # Pass the file object directly
                    results = resume_rag_main(
                        pdf_file=uploaded_file_instance, 
                        target_candidate_name=st.session_state.user_name,
                        force_recreate=True # Force recreate as it's a new file or re-upload
                    )
                if results and results.get('results'): # Check for 'results' key from RAG output
                    st.session_state.rag_results = results['results']
                    st.success(f"Resume '{uploaded_file_instance.name}' processed and analyzed successfully!")
                    update_user_data_in_db({ # Update DB with file name and RAG results
                        'file_name': uploaded_file_instance.name,
                        # RAG results are handled by update_user_data_in_db internally
                        'status': 'resume_uploaded' 
                    })
                else:
                    st.error("Failed to process the resume. The RAG system did not return expected results.")
                    st.session_state.uploaded_file_details = None # Clear if processing failed
            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")
                st.session_state.uploaded_file_details = None # Clear on error
    
    # Display RAG results if available
    if st.session_state.rag_results:
        with st.expander("📄 Resume Analysis Preview", expanded=False):
            for res_idx, res_item in enumerate(st.session_state.rag_results):
                st.markdown(f"**Query:** {res_item['query']}")
                st.markdown(f"**Answer:** {res_item['answer']}")
                if res_idx < len(st.session_state.rag_results) - 1:
                    st.markdown("---")

    # Form completion check
    is_form_complete = len(current_selection) > 0 and st.session_state.uploaded_file_details is not None and st.session_state.rag_results is not None
    
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    col_back, col_submit = st.columns(2)
    with col_back:
        if st.button("← Back to User Details"):
            st.session_state.selected_professors = current_selection # Save current selection
            st.session_state.page = 2
            st.rerun()
    with col_submit:
        submit_disabled = not is_form_complete
        if st.button("Generate Emails ✨", disabled=submit_disabled, type="primary"):
            st.session_state.selected_professors = current_selection
            
            # Final update to DB before generation
            update_user_data_in_db({
                'professors': current_selection,
                'status': 'submitted_for_generation'
                # resume_analysis and file_name should already be in DB from file upload logic
            })

            try:
                with st.spinner("Generating personalized emails... This may take a moment."):
                    # Pass a copy of st.session_state to avoid direct modification issues if any
                    current_session_data_for_agent = st.session_state.to_dict()
                    email_system = EmailAgentSystem(current_session_data_for_agent) # Pass full session_state
                    
                    email_generation_result = asyncio.run(email_system.generate_personalized_email())
                    
                if email_generation_result.get('status') == 'success':
                    st.session_state.generated_emails = email_generation_result.get('emails', {})
                    st.success("Emails generated successfully!")
                    st.balloons()
                    # Rerun to display emails, or display them directly here.
                    # For simplicity, let's store and let the next block handle display.
                else:
                    st.error(f"Error during email generation: {email_generation_result.get('error', 'Unknown error')}")
                    st.session_state.generated_emails = None # Clear on error
            except Exception as e:
                st.error(f"An unexpected error occurred in the email generation process: {str(e)}")
                st.exception(e) # Shows traceback for debugging
                st.session_state.generated_emails = None

    st.markdown('</div>', unsafe_allow_html=True) # Close button-row
            
    # Display generated emails if available
    if 'generated_emails' in st.session_state and st.session_state.generated_emails:
        st.markdown("---")
        st.subheader("Generated Email Previews:")
        for prof_name, email_details in st.session_state.generated_emails.items():
            with st.expander(f"Email for {prof_name}", expanded=False):
                if email_details.get('status') == 'success' and 'email' in email_details:
                    actual_email = email_details['email']
                    
                    # Use text_input and text_area with unique keys and store in variables
                    subject = st.text_input(
                        "Subject:", 
                        value=actual_email.get('subject', 'N/A'), 
                        key=f"subject_{prof_name}"
                    )
                    
                    body = st.text_area(
                        "Body:", 
                        value=actual_email.get('body', 'N/A'), 
                        height=300, 
                        key=f"email_body_{prof_name}"
                    )
                    
                    # Use the modified content for download and sending
                    email_text_content = f"Subject: {subject}\n\n{body}"
                    
                    # Create two columns for the buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label=f"Download Email for {prof_name}",
                            data=email_text_content,  # This will use the modified content
                            file_name=f"email_to_{prof_name.lower().replace(' ', '_')}.txt",
                            mime="text/plain",
                            key=f"download_{prof_name}"
                        )
                    
                    with col2:
                        if st.button(f"Send Email to {prof_name}", key=f"send_{prof_name}"):
                            try:
                                # Get professor's email from CSV
                                csv_path = os.path.join(os.path.dirname(__file__), 
                                    f'/Users/ShahYash/Desktop/Projects/ask-my-prof/AskMyProf/data/prof_data/{sanitize_university_name(st.session_state.university)}.csv')
                                
                                prof_df = pd.read_csv(csv_path)
                                prof_email = prof_df[prof_df['Name'] == prof_name]['email'].iloc[0]
                                
                                if not prof_email or pd.isna(prof_email):
                                    st.error(f"Could not find email address for {prof_name}")
                                    continue
                                
                                # Construct the email_sender.py path
                                email_sender_path = os.path.join(
                                    os.path.dirname(__file__),
                                    'email_sender.py'
                                )
                                
                                # Create environment variables for email_sender.py using modified content
                                env = os.environ.copy()
                                env.update({
                                    'EMAIL_SUBJECT': subject,  # Using modified subject
                                    'EMAIL_BODY': body,        # Using modified body
                                    'SENDER_EMAIL': st.session_state.user_email,
                                    'SENDER_PASSWORD': st.session_state.user_name,
                                    'RECIPIENT_EMAIL': prof_email,
                                    'RECIPIENT_NAME': prof_name
                                })
                                
                                # Run email_sender.py with the environment variables
                                with st.spinner("Sending email..."):
                                    process = subprocess.run(
                                        ['python3', email_sender_path],
                                        env=env,
                                        capture_output=True,
                                        text=True
                                    )
                                    
                                    if process.returncode == 0:
                                        st.success(f"Email sent successfully to {prof_name} at {prof_email}")
                                    else:
                                        error_msg = process.stderr or "Unknown error"
                                        st.error(f"Failed to send email: {error_msg}")
                                        if "authentication failed" in error_msg.lower():
                                            st.info("""
                                            Please ensure:
                                            1. Your email and password are correct
                                            2. You have enabled 'Less secure app access' in your Google Account
                                            3. Or use an App Password if you have 2-factor authentication enabled
                                            """)
                                
                            except Exception as e:
                                st.error(f"Error preparing to send email: {str(e)}")

                else:
                    error_msg = email_details.get('error', 'Unknown error during generation for this professor.')
                    st.error(f"Could not generate email for {prof_name}. Error: {error_msg}")

    # Informational messages if form is not complete for submission
    if not is_form_complete:
        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
        if not current_selection:
            st.markdown('<p class="info-message">📚 Please select at least one professor.</p>', unsafe_allow_html=True)
        if st.session_state.uploaded_file_details is None:
            st.markdown('<p class="info-message">📄 Please upload your resume.</p>', unsafe_allow_html=True)
        elif st.session_state.rag_results is None: # File uploaded but not processed
             st.markdown('<p class="info-message">🔄 Resume processing pending or failed. Please re-upload if necessary.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# Page Router
if st.session_state.page == 1:
    page_one()
elif st.session_state.page == 2:
    page_two()
else: # Page 3 or beyond (if you add more steps)
    page_three()

# Ensure database connection is closed when the app exits
def cleanup_db():
    print("Closing database connection...") # For console log
    db.close_connection()

# Register the cleanup function
atexit.register(cleanup_db)