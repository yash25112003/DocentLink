import streamlit as st
import os
import sys

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set page configuration
st.set_page_config(
    page_title="TA/RA Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import home page module
from pages.home import show_home_page

def main():
    """
    Main application entry point
    """
    # Create sidebar navigation
    with st.sidebar:
        st.title("🧠 TA/RA Assistant")
        st.markdown("---")
        
        # Navigation menu
        page = st.radio(
            "Navigation",
            ["Home", "Email Preview", "About"],
            index=0
        )
        
        st.markdown("---")
        
        # App information
        st.markdown("### About this app")
        st.info("""
        This tool helps students create personalized emails to professors
        for TA and RA positions by analyzing both the student's resume
        and the professor's research interests.
        """)
        
        # Footer
        st.markdown("---")
        st.caption("© 2025 TA/RA Assistant")

    # Display the selected page
    if page == "Home":
        show_home_page()
    elif page == "Email Preview":
        st.title("📧 Email Preview")
        
        if not st.session_state.get('email_generation_requested'):
            st.info("No email has been generated yet. Please go to the Home page to upload your resume and enter professor information.")
        else:
            st.warning("Email generation feature is under development.")
            
            # Placeholder for email preview
            st.markdown("""
            ### Sample Email Preview
            
            *This is a placeholder for the generated email*
            
            ---
            
            **Subject:** Application for Research Assistant Position
            
            Dear Professor Smith,
            
            I am a Master's student in Computer Science at Example University, and I am writing to express my interest in the Research Assistant position in your NLP for Healthcare project.
            
            [Rest of email would appear here...]
            
            ---
            
            ### Skill Recommendations
            
            Based on the professor's research, we recommend developing these skills:
            1. Natural Language Processing techniques
            2. Healthcare data analysis
            3. Python libraries for medical text processing
            """)
    
    elif page == "About":
        st.title("ℹ️ About TA/RA Assistant")
        st.markdown("""
        ## Project Overview
        
        This project is a **smart assistant tool** designed to help students—especially those applying abroad—**automatically generate personalized, high-quality emails** to university professors when inquiring about **TA (Teaching Assistant)** or **RA (Research Assistant)** positions.
        
        ### Key Features
        
        1. **Resume Analysis**
           * Extracts your skills, education, experiences, and research background
        
        2. **Professor Profile Analysis**
           * Analyzes the professor's research interests, projects, and required skills
        
        3. **Personalized Email Generation**
           * Creates a custom email highlighting alignment between your profile and the professor's work
        
        4. **Skill Recommendations**
           * Suggests skills you should develop to improve your chances
        
        ### How It Works
        
        1. Upload your resume
        2. Enter professor information
        3. We analyze both and generate a personalized email
        4. Review, edit, and use the email to contact the professor
        """)

if __name__ == "__main__":
    main()