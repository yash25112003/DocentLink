import streamlit as st
import os
import pandas as pd
from typing import List, Optional

# Cache the professor data loading
@st.cache_data
def load_professor_data(university: str) -> List[str]:
    """
    Load professor data from CSV file with caching.
    Future enhancement: Replace with database integration
    """
    file_path = f"/Users/ShahYash/Desktop/Projects/ask-my-prof/AskMyProf/data/prof_data/{sanitize_university_name(university)}"
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'Name' in df.columns:
                return df['Name'].tolist()
            else:
                st.warning(f"Unexpected format in {file_path}. Falling back to default professor names.")
        else:
            st.warning(f"No professor data available for {university}. Falling back to default professor names.")
    except Exception as e:
        st.error(f"Error loading professor data: {e}")
    
    # Return default professors
    return ["Dr. Smith", "Prof. Johnson", "Dr. Brown", "Prof. Davis", "Dr. Wilson"]

def sanitize_university_name(name: str) -> str:
    """Convert university name to lowercase with underscores for file naming"""
    return name.lower().replace(" ", "_") + ".csv"

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
    
    # Save the uploaded file to session state
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
    
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
            if uploaded_file:
                # Process file contents if needed
                print("Form Submitted!")  # Console log for verification
    
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
        
# Page Router
if st.session_state.page == 1:
    page_one()
elif st.session_state.page == 2:
    page_two()
else:
    page_three()