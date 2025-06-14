# 📚 DocentLink: AI-Powered Academic Outreach

**Connect. Apply. Succeed.**

DocentLink is an AI-driven platform designed to automate and personalize academic outreach, enabling a streamlined "one-step application" process. It empowers users to connect with professors by intelligently matching resumes with academic profiles and generating tailored emails.


## 🚀 Features

* **One-Step Application**: Streamlined workflow for uploading a resume and generating personalized outreach emails.
* **Intelligent Resume Analysis (RAG)**: Processes PDF resumes to extract key skills, experiences, projects, and contact information, making it queryable.
* **Dynamic Professor Profile Scraping**: Gathers detailed academic information (research interests, publications, teaching, contact) from university webpages using multiple robust scraping methods (Firecrawl, Playwright, Browser-Use).
* **Personalized Email Generation**: Utilizes advanced AI models to compose unique, relevant emails tailored to both the user's resume and the professor's specific academic profile.
* **User-Friendly Interface**: An intuitive Streamlit-based frontend for seamless interaction.
* **Secure Email Delivery**: Integrates with SMTP for direct email sending, complete with editable subjects and bodies.
* **Persistent Data Storage**: Stores user profiles and processed data using MongoDB for a continuous experience.
* **Schema-Driven Data Extraction**: Dynamically generates JSON schemas for web-scraped content to ensure accurate and comprehensive data capture.
* **Multi-Agent Architecture**: Employs Google ADK to orchestrate specialized agents for efficient task handling.


## 🔧 Technologies Used

### 💻 Frontend:

* Streamlit (for interactive web interface)

### ⚙️ Backend & Core Logic:

* Python
* Google Gemini API (for LLM interactions, email generation, schema generation, data extraction)
* Qdrant (Vector database for RAG, storing resume embeddings)
* PyMuPDF (fitz) (for PDF processing and text extraction)
* Sentence Transformers (for generating text embeddings for RAG)
* Firecrawl (Web scraping API)
* Playwright (Headless browser automation for advanced web scraping)
* Browser-Use (Agentic browser automation for web content extraction)
* BeautifulSoup4 (HTML parsing)
* Requests (HTTP requests)
* Langchain (Integration with LLMs, especially langchain-google-genai)
* Groq LLM (Potentially used for faster RAG responses, alongside Gemini)
* MongoDB (NoSQL database for user profiles and structured data)
* python-dotenv (Environment variable management)
* SMTP (for sending emails)
* subprocess (for running email sender as a separate process)
* asyncio and nest-asyncio (for asynchronous operations)
* Google ADK (Agent Development Kit - for implementing multi-agent systems)


## 🔀 Multi-Agent System (via Google ADK)

### ✨ Orchestrator Agent

* Coordinates interactions among all agents
* Maintains session state
* Implements fallback and retry mechanisms
* Enforces prompt safety and flow control

### 🔍 Professor Data Agent

* Parses and extracts professor JSON files
* Summarizes research interests, papers, awards, and affiliations

### 📁 Database Agent

* Fetches and stores user metadata via MongoDB
* Structures and normalizes resume data

### 🌐 LLM Agent

* Calls Gemini Pro to generate emails with 300–350 words
* Maintains formal academic tone
* Ensures reference to professor's work and user's background alignment


## ⚖️ Setup and Installation

### 1. Clone the Repository

```bash
git clone <repository_url>  # Replace with your actual repository URL
cd DocentLink

Install dependencies:

pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file
GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"
QDRANT_URL="YOUR_QDRANT_CLOUD_URL"
QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
GROQ_API_KEY="YOUR_GROQ_API_KEY"
FIRE_CRAWL_API_KEY="YOUR_FIRECRAWL_API_KEY"
MONGODB_URI="YOUR_MONGODB_CONNECTION_STRING"
BRIGHTDATA_WSS_URL="YOUR_BRIGHTDATA_PROXY_WSS_URL"
```

3. Create a Virtual Environment
``` bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

```
4. Install Dependencies
``` bash
pip install -r requirements.txt
```
If no requirements.txt, install manually:
``` bash
pip install streamlit pymongo python-dotenv google-generativeai qdrant-client sentence-transformers PyMuPDF firecrawl-py "playwright>=1.40" beautifulsoup4 requests langchain-google-genai google-adk
playwright install
```
5. Use Professor Data from Location

The application expects professor data in CSV files located within the **data/prof_data directory** at the project root.

## Usage

1. Run the Streamlit app:
```bash
streamlit run frontend/pages/home.py
```

2. Fill out the form with:
   - University selection
   - User email and Password
   - Professor selection
   - Resume upload

3. Genrate and send personalized emails

   - Click "Generate Emails ✨".
   - Edit subject/body as needed.
   - Download or send via SMTP directly.

## Project Structure
``` bash

  DocentLink/
  ├── backend/
  │   ├── agents/
  │   │   └── email_agent_system.py         # ADK-based email generation agent system
  │   ├── db_manager.py                    # MongoDB interface
  │   └── scrapper/
  │       ├── config.py                    # Config & keys
  │       ├── llm_handler.py               # LLM API interactions
  │       ├── main.py                      # End-to-end pipeline runner
  │       ├── processing.py                # Data cleanup and transformation
  │       ├── schema_generator.py         # Dynamic schema generator
  │       └── scraping_manager.py         # Orchestrates scrapers
  ├── data/
  │   └── prof_data/                       # CSVs for professor info
  ├── frontend/
  │   ├── pages/
  │   │   └── home.py                      # Streamlit UI
  │   └── rag/
  │       ├── resume_rag.py                # Resume RAG pipeline
  │       └── test_resumes/               # Sample resumes
  ├── email_sender.py                     # SMTP email sender script
  ├── .env                                # Environment configuration
  ├── README.md                           # This file
  └── requirements.txt                    # Python dependencies

```
