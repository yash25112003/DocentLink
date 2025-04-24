import streamlit as st
import os
import sys
import tempfile
import base64
from datetime import datetime

def show_home_page():
    """
    Main function to display the home page with resume upload and professor information input
    """
    # Initialize session state variables if they don't exist
    if 'resume_path' not in st.session_state:
        st.session_state['resume_path'] = None
    if 'resume_filename' not in st.session_state:
        st.session_state['resume_filename'] = None
    if 'professor_data' not in st.session_state:
        st.session_state['professor_data'] = None
    if 'parsed_resume_data' not in st.session_state:
        st.session_state['parsed_resume_data'] = None

    # Page header with styling
    st.title("🧠 TA/RA Application Assistant")
    st.markdown("""
    <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:20px;">
    <h4>Generate personalized emails to professors for Teaching Assistant (TA) or Research Assistant (RA) positions</h4>
    <p>Upload your resume and provide the professor's information to get started.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📄 Resume Upload", "🎓 Professor Information", "📋 Summary"])

    with tab1:
        handle_resume_upload()

    with tab2:
        handle_professor_information()

    with tab3:
        display_summary()

    # Action buttons at the bottom of the page
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear All Data", type="secondary"):
            st.session_state['resume_path'] = None
            st.session_state['resume_filename'] = None
            st.session_state['professor_data'] = None
            st.session_state['parsed_resume_data'] = None
            st.rerun()
    
    with col2:
        if st.session_state.get('resume_path') and st.session_state.get('professor_data'):
            if st.button("Generate Email", type="primary"):
                # This would normally trigger the email generation process
                st.session_state['email_generation_requested'] = True
                st.success("Email generation requested!")
                st.info("Processing professor's website and generating your personalized email...")
                # In a full implementation, you would redirect to the preview page
                st.markdown("You'll be redirected to the preview page when the email is ready.")


def handle_resume_upload():
    """
    Handle the resume upload functionality
    """
    st.header("Upload Your Resume")
    st.write("Upload your resume in PDF or DOCX format. We'll extract your skills, education, and experiences.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose your resume file", 
        type=["pdf", "docx"],
        help="We support PDF and DOCX formats."
    )

    if uploaded_file is not None:
        # Display file details
        col1, col2 = st.columns(2)
        with col1:
            st.write("📁 **File Details:**")
            st.write(f"- **Name**: {uploaded_file.name}")
            st.write(f"- **Type**: {uploaded_file.type}")
            st.write(f"- **Size**: {uploaded_file.size / 1024:.2f} KB")
        
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name
        
        # Store the file path and name in session state
        st.session_state['resume_path'] = temp_path
        st.session_state['resume_filename'] = uploaded_file.name
        
        # Show success message with option to view file
        with col2:
            st.success(f"Resume uploaded successfully!")
            
            # Create a download button to verify the uploaded file
            file_bytes = uploaded_file.getvalue()
            b64 = base64.b64encode(file_bytes).decode()
            href = f'<a href="data:{uploaded_file.type};base64,{b64}" download="{uploaded_file.name}" target="_blank">View uploaded resume</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # In a full implementation, we would parse the resume here
        st.info("Resume parsing would extract your details here. In this version, we're just storing the file.")
        
        # Mock parsed data for demonstration
        st.session_state['parsed_resume_data'] = {
            "name": "Sample Student",
            "education": [
                {"degree": "MS in Computer Science", "institution": "Example University", "year": "2022-2024"}
            ],
            "skills": ["Python", "Machine Learning", "Data Analysis", "Web Development"],
            "experience": [
                {"title": "Research Intern", "company": "Tech Lab", "duration": "Summer 2023"}
            ]
        }


def handle_professor_information():
    """
    Handle the professor information input form
    """
    st.header("Professor Information")
    st.write("Enter details about the professor you want to contact for a TA/RA position.")

    # Create a form for professor information
    with st.form("professor_form"):
        prof_name = st.text_input("Professor's Full Name *", 
                                  value=st.session_state.get('professor_data', {}).get('professor_name', ''),
                                  placeholder="e.g., Dr. Jane Smith")
        
        university = st.text_input("University Name *", 
                                  value=st.session_state.get('professor_data', {}).get('university', ''),
                                  placeholder="e.g., Stanford University")
        
        department = st.text_input("Department *", 
                                 value=st.session_state.get('professor_data', {}).get('department', ''),
                                 placeholder="e.g., Computer Science")
        
        website_url = st.text_input("Professor's Website URL *", 
                                   value=st.session_state.get('professor_data', {}).get('website_url', ''),
                                   placeholder="e.g., https://cs.stanford.edu/~jsmith")
        
        position_type = st.selectbox("Position Type *", 
                                    options=["Research Assistant (RA)", 
                                             "Teaching Assistant (TA)", 
                                             "Both RA and TA"],
                                    index=0 if not st.session_state.get('professor_data') else 
                                         ["Research Assistant (RA)", "Teaching Assistant (TA)", "Both RA and TA"].index(
                                             st.session_state.get('professor_data', {}).get('position_type', "Research Assistant (RA)")
                                         ))
        
        col1, col2 = st.columns(2)
        with col1:
            specific_course = st.text_input("Specific Course (for TA)",
                                         value=st.session_state.get('professor_data', {}).get('specific_course', ''),
                                         placeholder="e.g., Machine Learning CS229")
        
        with col2:
            specific_project = st.text_input("Specific Research Project (for RA)",
                                          value=st.session_state.get('professor_data', {}).get('specific_project', ''),
                                          placeholder="e.g., NLP for Healthcare")
        
        additional_notes = st.text_area("Additional Notes (Optional)", 
                                       value=st.session_state.get('professor_data', {}).get('additional_notes', ''),
                                       placeholder="Any specific interests or information you want to highlight")
        
        # Form submission
        submit_button = st.form_submit_button("Save Professor Information", use_container_width=True)
        
        if submit_button:
            # Validate required fields
            if not prof_name or not university or not department or not website_url:
                st.error("Please fill in all required fields marked with *")
            else:
                # Store the data in session state
                st.session_state['professor_data'] = {
                    "professor_name": prof_name,
                    "university": university,
                    "department": department,
                    "website_url": website_url,
                    "position_type": position_type,
                    "specific_course": specific_course,
                    "specific_project": specific_project,
                    "additional_notes": additional_notes,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.success("Professor information saved successfully!")


def display_summary():
    """
    Display a summary of the resume and professor information
    """
    st.header("Summary")
    
    if not st.session_state.get('resume_path') and not st.session_state.get('professor_data'):
        st.info("No data available yet. Please upload your resume and enter professor information.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resume Information")
        if st.session_state.get('resume_path'):
            st.success(f"✅ Resume uploaded: {st.session_state.get('resume_filename')}")
            
            # Display parsed resume data if available
            if st.session_state.get('parsed_resume_data'):
                data = st.session_state['parsed_resume_data']
                st.write(f"**Name:** {data['name']}")
                
                st.write("**Education:**")
                for edu in data['education']:
                    st.write(f"- {edu['degree']} at {edu['institution']} ({edu['year']})")
                
                st.write("**Skills:**")
                st.write(", ".join(data['skills']))
                
                st.write("**Experience:**")
                for exp in data['experience']:
                    st.write(f"- {exp['title']} at {exp['company']} ({exp['duration']})")
        else:
            st.warning("⚠️ No resume uploaded yet")
    
    with col2:
        st.subheader("Professor Information")
        if st.session_state.get('professor_data'):
            data = st.session_state['professor_data']
            st.success("✅ Professor information saved")
            st.write(f"**Name:** {data['professor_name']}")
            st.write(f"**University:** {data['university']}")
            st.write(f"**Department:** {data['department']}")
            st.write(f"**Position Type:** {data['position_type']}")
            
            if data['position_type'] == "Teaching Assistant (TA)" or data['position_type'] == "Both RA and TA":
                if data['specific_course']:
                    st.write(f"**Course:** {data['specific_course']}")
            
            if data['position_type'] == "Research Assistant (RA)" or data['position_type'] == "Both RA and TA":
                if data['specific_project']:
                    st.write(f"**Research Project:** {data['specific_project']}")
        else:
            st.warning("⚠️ No professor information saved yet")
            