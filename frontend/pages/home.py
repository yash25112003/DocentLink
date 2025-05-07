import streamlit as st
import os
import pandas as pd
from typing import List, Optional
import uuid
from datetime import datetime
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Now you can import
from frontend.rag.resume_rag import resume_rag_main

# Add the backend directory to Python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend'))
sys.path.append(backend_path)

# Import the DatabaseManager
from db_manager import DatabaseManager

# Initialize database manager
db = DatabaseManager()

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
        csv_path = os.path.join(os.path.dirname(__file__), f'../../data/prof_data/{safe_university}.csv')
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'Name' in df.columns:
                return df['Name'].tolist()
            else:
                st.warning(f"CSV file for {university} exists but doesn't have a 'name' column")
                return []
        else:
            st.warning(f"No CSV file found for {university} at {csv_path}")
            return []
            
    except Exception as e:
        st.error(f"Error loading professor data: {str(e)}")
        return []

def sanitize_university_name(name: str) -> str:
    """Convert university name to lowercase with underscores for file naming"""
    return name.lower().replace(" ", "_")

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
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    # Create a new user entry in MongoDB
    user_data = {
        'user_id': st.session_state.user_id,
        'name': None,
        'email': None,
        'university': None,
        'professors': [],
        'file_name': None,
        'status': 'incomplete',
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }
    db.create_user_profile(user_data)

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
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Common title across pages
st.markdown("<div class='main-title'>AskMyProf</div>", unsafe_allow_html=True)

def update_user_data(update_fields: dict):
    """Helper function to update user data in MongoDB"""
    update_fields['updated_at'] = datetime.utcnow()
    db.update_user_profile(st.session_state.user_id, update_fields)

def page_one():
    """University Selection Page"""
    with st.form(key="page_one_form"):
        st.subheader("University Selection")
        university = st.selectbox(
            "Select your university:",
            ["Select University", "Arizona State University", "Stanford University", "MIT", "Harvard University", "UC Berkeley"],
            index=["Select University", "Arizona State University", "Stanford University", "MIT", "Harvard University", "UC Berkeley"].index(st.session_state.university)
        )

        submitted = st.form_submit_button("Continue to User Details")
        
        if submitted:
            if university == "Select University":
                st.error("Please select a university to continue.")
                return
            
            # Reset submission state and selected professors if university changed
            if st.session_state.university != university:
                st.session_state.submission_success = False
                st.session_state.selected_professors = []
                
            # Update user profile in MongoDB
            update_user_data({
                'university': university,
                'status': 'university_selected'
            })
                
            # Store values in session state
            st.session_state.university = university
            st.session_state.page = 2
            st.rerun()

def page_two():
    """User Details Page"""
    with st.form(key="page_two_form"):
        st.subheader("User Details")
        user_name = st.text_input(
            "User Name",
            value=st.session_state.user_name,
            placeholder="Enter your name"
        )
        user_email = st.text_input(
            "User Email",
            value=st.session_state.user_email,
            placeholder="Enter your email"
        )

        col1, col2 = st.columns(2)
        with col1:
            back_button = st.form_submit_button("← Back")
        with col2:
            continue_button = st.form_submit_button("Continue to Professor Selection")

        if back_button:
            # Store current values before navigating back
            st.session_state.user_name = user_name
            st.session_state.user_email = user_email
            st.session_state.page = 1
            st.rerun()
            
        if continue_button:
            if not user_name.strip():
                st.error("Please enter your name.")
                return
            if not user_email or "@" not in user_email or "." not in user_email:
                st.error("Please enter a valid email address.")
                return
            
            # Update user profile in MongoDB
            update_user_data({
                'name': user_name,
                'email': user_email,
                'status': 'details_completed'
            })
            
            # Store values in session state
            st.session_state.user_name = user_name
            st.session_state.user_email = user_email
            st.session_state.page = 3
            st.rerun()

def page_three():
    """Professor Selection & File Upload Page"""
    st.subheader(f"Professor Selection for {st.session_state.university}")
    
    # Add container styling
    st.markdown("""
    <style>
    .button-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    .stButton > button {
        border-radius: 5px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .back-button > button {
        background-color: #f1f3f4;
        color: #3c4043;
        border: 1px solid #dadce0;
    }
    .back-button > button:hover {
        background-color: #e8eaed;
    }
    .submit-button > button {
        background-color: #4285f4;
        color: white;
        border: none;
    }
    .submit-button > button:hover {
        background-color: #3b78e7;
    }
    .info-message {
        padding: 10px 15px;
        border-radius: 5px;
        margin-top: 0.3rem;
        background-color: #e8f0fe;
        border-left: 3px solid #4285f4;
        font-size: 0.9rem;
    }
    .success-message {
        background-color: #e6f4ea;
        color: #137333;
        padding: 15px;
        border-radius: 5px;
        margin-top: 1rem;
        text-align: center;
        font-weight: 500;
        border-left: 3px solid #137333;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Start form container
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # Load professors with loading spinner
    with st.spinner("Loading professor data..."):
        professors_list = load_professor_data(st.session_state.university)
    
    # Reset selected professors if they aren't in the current list
    valid_selections = [p for p in st.session_state.selected_professors if p in professors_list]
    st.session_state.selected_professors = valid_selections
    
    # Store the current selection in a variable
    current_selection = st.multiselect(
        "Select up to 10 professors:",
        options=professors_list,
        max_selections=10,
        default=valid_selections
    )
    
    st.markdown('<hr style="margin: 0.3rem 0; border-color: #f0f0f0;">', unsafe_allow_html=True)
    
    st.subheader("File Upload")
    uploaded_file = st.file_uploader(
        "Upload a file (PDF, DOCX, TXT):",
        type=["pdf", "docx", "txt"],
        key="file_uploader"
    )
    
    # Handle file changes
    if uploaded_file is not None:
        if st.session_state.uploaded_file != uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            # Process the file directly in memory
            try:
                results = resume_rag_main(
                    pdf_file=uploaded_file,
                    target_candidate_name=st.session_state.user_name,  # Optional: use user's name if available
                    force_recreate=False
                )
                if results and results.get('resume_rag') and results.get('candidate'):
                    st.session_state.rag_system = results['resume_rag']
                    st.session_state.candidate_map = {results['candidate'].get('name', 'unknown'): results['candidate']}
                    st.session_state.rag_results = results['results']  # Store the query results if you want to display them
                    st.success("File processed and analyzed successfully!")
                else:
                    st.error("Failed to process the file. Please try again.")
                    st.session_state.uploaded_file = None
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.session_state.uploaded_file = None
    else:
        if st.session_state.uploaded_file is not None:
            st.session_state.uploaded_file = None
            st.session_state.rag_system = None
            st.session_state.candidate_map = None
            # Update database to remove file reference
            update_user_data({
                'file_name': None,
                'status': 'file_removed'
            })
    
    # Check if form is complete
    is_form_complete = len(current_selection) > 0 and uploaded_file is not None
    
    # Success state to track submission
    if 'submission_success' not in st.session_state:
        st.session_state.submission_success = False
    
    # Button row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="back-button">', unsafe_allow_html=True)
        if st.button("← Back"):
            st.session_state.selected_professors = current_selection
            st.session_state.page = 2
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="submit-button">', unsafe_allow_html=True)
        submit_button = st.button("Submit", disabled=not is_form_complete)
        st.markdown('</div>', unsafe_allow_html=True)
        if submit_button and is_form_complete:
            st.session_state.selected_professors = current_selection
            st.session_state.submission_success = True
            
            # Update user data in MongoDB
            update_user_data({
                'professors': current_selection,
                'file_name': uploaded_file.name if uploaded_file else None,
                'status': 'submitted'
            })
    
    # Close the form container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display success message AFTER form container but before info messages
    if st.session_state.submission_success:
        st.markdown('<div class="success-message">Form submitted successfully!</div>', unsafe_allow_html=True)
        st.balloons()  # Add a fun effect on successful submission
    
    # Display info messages in their own container
    if not current_selection or not uploaded_file:
        st.markdown('<div class="form-container" style="padding: 0;">', unsafe_allow_html=True)
        if not current_selection:
            st.markdown('<div class="info-message">📚 Please select at least one professor.</div>', unsafe_allow_html=True)
        
        if not uploaded_file:
            st.markdown('<div class="info-message">📄 Please upload a file to continue.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    if 'rag_results' in st.session_state and st.session_state.rag_results:
        st.subheader("Resume Analysis Results")
        for res in st.session_state.rag_results:
            st.markdown(f"**Q:** {res['query']}")
            st.markdown(f"**A:** {res['answer']}")
            st.markdown("---")

# Page Router
if st.session_state.page == 1:
    page_one()
elif st.session_state.page == 2:
    page_two()
else:
    page_three()

# Ensure database connection is closed when the app exits
def cleanup():
    db.close_connection()

# Register the cleanup function
import atexit
atexit.register(cleanup)