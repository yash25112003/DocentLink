import os
import json
import re
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from firecrawl import FirecrawlApp, JsonConfig
from pydantic import BaseModel, Field
from google import genai

# Load environment variables
load_dotenv()

# Initialize clients
firecrawl_api_key = os.getenv("FIRE_CRAWL_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not firecrawl_api_key:
    raise ValueError("FIRE_CRAWL_API_KEY not found in environment variables")

app = FirecrawlApp(api_key=firecrawl_api_key)
gemini_client = genai.Client(api_key=gemini_api_key)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WebScraper")

class ProfessorInfo(BaseModel):
    """Schema to extract professor's basic information"""
    professor_name: str = Field(..., description="Full name of the professor")
    position: Optional[str] = Field(None, description="Academic position or title")
    department: Optional[str] = Field(None, description="Department or faculty")
    university: Optional[str] = Field(None, description="University or institution")
    email: Optional[str] = Field(None, description="Professional email address")
    research_interests: Optional[List[str]] = Field(None, description="List of research interests")
    website_url: Optional[str] = Field(None, description="Main website URL")

class WebsiteContent(BaseModel):
    """Schema for website content sections"""
    section_title: str = Field(..., description="Title of the content section")
    content: str = Field(..., description="Markdown content of the section")
    url: str = Field(..., description="URL where this content was found")

class WebsiteExtractionResult(BaseModel):
    """Final structured output schema"""
    professor_info: ProfessorInfo
    content_sections: List[WebsiteContent]
    internal_links: List[str]
    external_links: List[str]

def extract_professor_name(url: str) -> Optional[str]:
    """Extract professor's name from the website using Gemini"""
    try:
        # First try Firecrawl's built-in extraction
        extract_config = JsonConfig(schema=ProfessorInfo.model_json_schema())
        scrape_result = app.scrape_url(url, params={'jsonOptions': extract_config})
        
        if scrape_result and scrape_result.get('json'):
            return scrape_result['json'].get('professor_name')
        
        # Fallback to Gemini if Firecrawl extraction fails
        logger.info("Using Gemini to extract professor name...")
        prompt = f"""
        Extract the full name of the professor or primary individual this academic website belongs to.
        Website URL: {url}
        Return ONLY the name in plain text format, nothing else.
        """
        
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        name = response.text.strip()
        if name and len(name.split()) >= 2:  # Basic validation that it looks like a name
            return name
        return None
        
    except Exception as e:
        logger.error(f"Error extracting professor name: {e}")
        return None

def extract_links_from_website(url: str) -> Dict[str, List[str]]:
    """Extract all internal and external links from a webpage"""
    try:
        # Use Firecrawl to get all links
        class LinkSchema(BaseModel):
            links: List[str] = Field(..., description="All links found on the page")
        
        extract_config = JsonConfig(schema=LinkSchema.model_json_schema())
        result = app.scrape_url(url, params={'jsonOptions': extract_config})
        
        if not result or not result.get('json'):
            return {'internal': [], 'external': []}
            
        all_links = result['json'].get('links', [])
        
        # Categorize links as internal or external
        domain = re.sub(r'https?://(www\.)?', '', url).split('/')[0]
        internal_links = []
        external_links = []
        
        for link in all_links:
            if not link.startswith(('http://', 'https://')):
                continue
            if domain in link:
                internal_links.append(link)
            else:
                external_links.append(link)
                
        return {
            'internal': list(set(internal_links)),  # Remove duplicates
            'external': list(set(external_links))
        }
        
    except Exception as e:
        logger.error(f"Error extracting links from {url}: {e}")
        return {'internal': [], 'external': []}

def extract_content_with_dynamic_schema(url: str) -> Dict:
    """Extract content from a webpage using dynamic schema based on headings"""
    try:
        # First get the markdown content
        scrape_result = app.scrape_url(url, formats=["markdown"])
        if not scrape_result or not scrape_result.markdown:
            return {}
            
        # Use Gemini to analyze the content and create a schema
        prompt = f"""
        Analyze the following academic website content and identify the main sections.
        For each section, extract:
        - Section title (from heading)
        - Content (text under the heading)
        
        Website URL: {url}
        Content:
        {scrape_result.markdown[:10000]}  # Limit to first 10k chars
        
        Return a JSON object with this structure:
        {{
            "sections": [
                {{
                    "title": "Section title",
                    "content": "Section content in markdown format",
                    "url": "{url}"
                }}
            ]
        }}
        """
        
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        # Clean and parse the response
        cleaned_response = response.text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[len('```json'):].strip()
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-len('```')].strip()
            
        return json.loads(cleaned_response)
        
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return {}

def crawl_website(main_url: str, max_pages: int = 20) -> WebsiteExtractionResult:
    """Crawl a professor's website and extract structured content"""
    logger.info(f"Starting crawl for {main_url}")
    
    # Step 1: Get professor's name
    professor_name = extract_professor_name(main_url)
    if not professor_name:
        professor_name = input("Could not extract professor name automatically. Please enter a name: ")
    
    # Step 2: Get initial content from main page
    main_content = extract_content_with_dynamic_schema(main_url)
    links = extract_links_from_website(main_url)
    
    # Step 3: Crawl internal links (up to max_pages)
    all_content = main_content.get('sections', [])
    crawled_urls = {main_url}
    
    for link in links['internal'][:max_pages]:
        if link not in crawled_urls:
            try:
                logger.info(f"Crawling internal link: {link}")
                content = extract_content_with_dynamic_schema(link)
                all_content.extend(content.get('sections', []))
                crawled_urls.add(link)
                
                # Get more links from this page
                new_links = extract_links_from_website(link)
                links['internal'].extend(new_links['internal'])
                links['external'].extend(new_links['external'])
                
            except Exception as e:
                logger.error(f"Error crawling {link}: {e}")
    
    # Step 4: Structure the final output
    professor_info = {
        'professor_name': professor_name,
        'website_url': main_url
    }
    
    # Try to extract more info from the main page
    try:
        extract_config = JsonConfig(schema=ProfessorInfo.model_json_schema())
        info_result = app.scrape_url(main_url, params={'jsonOptions': extract_config})
        if info_result and info_result.get('json'):
            professor_info.update(info_result['json'])
    except Exception as e:
        logger.warning(f"Could not extract additional professor info: {e}")
    
    result = {
        'professor_info': professor_info,
        'content_sections': all_content,
        'internal_links': list(set(links['internal'])),  # Remove duplicates
        'external_links': list(set(links['external']))
    }
    
    return WebsiteExtractionResult(**result)

def save_to_json(data: WebsiteExtractionResult, output_dir: str = "output"):
    """Save extraction results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from professor name
    sanitized_name = re.sub(r'[\\/*?:"<>|\s]+', '_', data.professor_info.professor_name)
    filename = f"{sanitized_name}_extracted.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data.model_dump(), f, indent=4, ensure_ascii=False)
    
    logger.info(f"Saved results to {filepath}")
    return filepath

def main():
    """Main function to run the scraper"""
    print("Academic Website Scraper")
    print("------------------------")
    url = input("Enter the professor's website URL: ").strip()
    
    if not url.startswith(('http://', 'https://')):
        print("Invalid URL. Please include http:// or https://")
        return
    
    try:
        result = crawl_website(url)
        saved_path = save_to_json(result)
        print(f"\nSuccess! Results saved to: {saved_path}")
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        print("An error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()