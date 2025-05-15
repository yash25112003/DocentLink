# AskMyProf - Academic Email Generation System

A sophisticated multi-agent system for generating personalized academic emails using Google's Agent Development Kit (ADK).

## Features

- Multi-agent system with specialized agents for:
  - Database operations
  - Professor data processing
  - Email generation using Gemini
- Automatic email generation based on:
  - User's academic background
  - Professor's research and publications
  - Specific alignment points between user and professor
- Professional email templates with:
  - Personalized subject lines
  - Formal academic tone
  - Specific references to professor's work
  - Clear connection between user's background and professor's research

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file
MONGODB_URI=your_mongodb_uri
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_credentials.json
```

3. Ensure professor data files are in the correct location:
```
backend/scrapper/output/
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run frontend/pages/home.py
```

2. Fill out the form with:
   - University selection
   - User details
   - Professor selection
   - Resume upload

3. Click Submit to generate personalized emails

## System Architecture

### Orchestrator Agent
- Manages workflow between sub-agents
- Implements error handling and fallback mechanisms
- Maintains session state
- Enforces safety guardrails

### Database Agent
- Connects to MongoDB
- Fetches user profile data
- Processes and structures user information

### Professor Data Agent
- Locates and processes professor JSON files
- Extracts research papers, projects, and achievements
- Handles multiple files per professor

### LLM Agent
- Uses Gemini model for email generation
- Ensures professional academic tone
- Maintains 300-350 word count
- Creates personalized content

## Error Handling

The system includes comprehensive error handling for:
- Missing professor files
- Incomplete user data
- LLM generation failures
- Database connection issues

## Quality Control

- Word count validation
- Professional tone enforcement
- Specific reference requirements
- Grammar and academic style checking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request