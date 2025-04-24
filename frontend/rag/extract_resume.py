import os
import uuid
import hashlib
import fitz  # PyMuPDF
import time
import re
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, SearchParams
from dotenv import load_dotenv
import warnings

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")

# --------------------------
# Configuration Models
# --------------------------
class ResumeConfig(BaseModel):
    candidate_name: str = ""  # Optional now, will be filled dynamically
    collection_prefix: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    text_vector_dim: int = 384
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
    chunk_size: int = 500

    @property
    def collection_name(self) -> str:
        """Generate consistent collection name from candidate name"""
        if not self.candidate_name:
            raise ValueError("Candidate name must be set before accessing collection_name")
        
        name_hash = hashlib.md5(self.candidate_name.strip().lower().encode()).hexdigest()
        return f"{self.collection_prefix}{name_hash}"

    class Config:
        """Pydantic config to allow .env field validation"""
        env_file = ".env"
        env_file_encoding = "utf-8"

# --------------------------
# Core Processing Classes
# --------------------------
class ResumeProcessor:
    """Handles PDF processing and text embedding"""
    
    def __init__(self, config: ResumeConfig):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model)
    
    def extract_candidate_name(self, file_bytes: bytes) -> str:
        """Extract candidate name from resume PDF"""
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            # Get text from first page where name is usually located
            first_page = doc[0].get_text()
            
            # Strategy 1: Look for common resume header patterns
            top_text = first_page.split('\n')[:10]  # First 10 lines
            
            # Clean up the text and look for potential names
            for line in top_text:
                line = line.strip()
                
                if not line or len(line) < 3 or line.lower() in ["resume", "curriculum vitae", "cv"]:
                    continue
                    
                if '@' in line or re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', line):
                    continue
                
                if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', line) or re.match(r'^[A-Z\s]+$', line):
                    print(f"✅ Extracted candidate name: {line}")
                    return line
            
            # Strategy 2: Look for common patterns like "Name: John Doe"
            name_pattern = re.search(r'(?:name|candidate):\s*([A-Za-z\s]+)', first_page, re.IGNORECASE)
            if name_pattern:
                name = name_pattern.group(1).strip()
                print(f"✅ Extracted candidate name: {name}")
                return name
            
            # Strategy 3: First line that looks like a name
            for line in top_text:
                line = line.strip()
                if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$', line):
                    print(f"✅ Extracted candidate name: {line}")
                    return line
            
            # If all else fails, use first non-empty line
            for line in top_text:
                if line.strip() and len(line.strip()) > 3:
                    print(f"⚠️ Using first line as candidate name: {line.strip()}")
                    return line.strip()
                    
            print("⚠️ Could not extract candidate name, using 'Unknown Candidate'")
            return "Unknown Candidate"
            
        except Exception as e:
            print(f"❌ Error extracting candidate name: {str(e)}")
            return "Unknown Candidate"
    
    def extract_from_pdf(self, file_bytes: bytes) -> List[Tuple[str, Dict]]:
    """Extract structured sections from resume PDF with smaller chunks"""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        sections = []
        
        for page in doc:
            text = page.get_text()
            if not text.strip():
                continue
                
            # Define section patterns
            section_types = [
                ("experience", ["experience", "work history", "employment", "professional experience"]),
                ("education", ["education", "academic background", "academic qualifications"]),
                ("skills", ["skills", "technical skills", "competencies", "expertise"]),
                ("projects", ["projects", "research projects", "personal projects"]),
                ("certifications", ["certifications", "certificates", "accreditations"]),
                ("languages", ["languages", "language proficiency"]),
                ("achievements", ["achievements", "accomplishments", "honors", "awards"]),
                ("summary", ["summary", "professional summary", "profile", "objective"])
            ]
            
            # Try to detect sections and chunk them
            section_chunks = {}
            current_section = "other"
            
            # Split by lines for better section detection
            lines = text.split('\n')
            section_start_line = 0
            
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                
                # Detect section headers
                for section_name, keywords in section_types:
                    if any(keyword in line_lower for keyword in keywords) and len(line) < 50:
                        # If we found a new section, add the previous one
                        if section_start_line < i:
                            section_text = '\n'.join(lines[section_start_line:i])
                            if section_text.strip():
                                if current_section not in section_chunks:
                                    section_chunks[current_section] = []
                                section_chunks[current_section].append(section_text)
                        
                        # Start a new section
                        current_section = section_name
                        section_start_line = i
                        break
            
            # Add the last section
            if section_start_line < len(lines):
                section_text = '\n'.join(lines[section_start_line:])
                if section_text.strip():
                    if current_section not in section_chunks:
                        section_chunks[current_section] = []
                    section_chunks[current_section].append(section_text)
            
            # Process sections into smaller chunks by paragraphs
            for section_type, contents in section_chunks.items():
                for content in contents:
                    # Split large content into smaller chunks
                    if len(content) > self.config.chunk_size:
                        chunks = []
                        paragraphs = content.split('\n\n')
                        current_chunk = ""
                        
                        for paragraph in paragraphs:
                            if len(current_chunk) + len(paragraph) < self.config.chunk_size:
                                current_chunk += paragraph + "\n\n"
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                current_chunk = paragraph + "\n\n"
                        
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        for chunk in chunks:
                            sections.append((section_type, {
                                "text": chunk,
                                "page": page.number,
                                "source": f"{section_type}_section"
                            }))
                    else:
                        sections.append((section_type, {
                            "text": content,
                            "page": page.number,
                            "source": f"{section_type}_section"
                        }))
        
        return sections if sections else [("full_resume", {
            "text": doc[0].get_text(),
            "page": 0,
            "source": "full_document"
        })]
    except Exception as e:
        print(f"❌ Error processing resume: {str(e)}")
        return []

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        return self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=8,
            convert_to_numpy=True
        ).tolist()

# --------------------------
# Vector Database Class
# --------------------------
class ResumeVectorDB:
    """Manages Qdrant operations with .env configuration"""
    
    def __init__(self, config: ResumeConfig):
        self.config = config
        self._validate_env_vars()
        self.client = self._initialize_client()
        self._initialize_collection()
    
    def _validate_env_vars(self):
        """Validate required environment variables"""
        if not self.config.qdrant_url:
            raise ValueError("QDRANT_URL not found in .env file")
        if not self.config.qdrant_api_key:
            print("⚠️ Warning: QDRANT_API_KEY not set - using unauthenticated connection")
    
    def _initialize_client(self) -> QdrantClient:
        """Initialize Qdrant client with .env configuration"""
        try:
            client = QdrantClient(
                url=self.config.qdrant_url, 
                api_key=self.config.qdrant_api_key if self.config.qdrant_api_key else None,
                timeout=120,
                prefer_grpc=True
            )
            print(f"✅ Qdrant client initialized at {self.config.qdrant_url}")
            return client
        except Exception as e:
            print(f"❌ Error initializing Qdrant client: {str(e)}")
            raise
    
    def _initialize_collection(self):
        """Create collection if it doesn't exist"""
        try:
            if not self.client.collection_exists(self.config.collection_name):
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.text_vector_dim,
                        distance=Distance.COSINE,
                        on_disk=True
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=0
                    )
                )
                print(f"✅ Created new collection: {self.config.collection_name}")
            else:
                print(f"ℹ️ Using existing collection: {self.config.collection_name}")
        except Exception as e:
            print(f"❌ Error initializing collection: {str(e)}")
            raise

    def check_document_exists(self, document_id: str) -> bool:
        """Check if a document already exists in Qdrant before inserting."""
        try:
            existing_docs, _ = self.client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                    ],  # This was missing the closing parenthesis
                ),  # Now properly closed
                limit=1
            )
            if existing_docs:
                print(f"⚠️ Document {document_id} already exists. Skipping ingestion.")
                return True
            return False
        except Exception as e:
            print(f"❌ Error checking document existence: {str(e)}")
            raise

    def ingest_resume(self, sections: List[Tuple[str, Dict]], embeddings: List[List[float]], max_retries: int = 3) -> bool:
        """Store resume data in Qdrant with enhanced metadata and retry logic"""
        document_id = f"resume_{int(time.time())}"
        
        if self.check_document_exists(document_id):
            return False
            
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "section_type": section_type,
                    "content": content["text"],
                    "page": content["page"],
                    "source": content.get("source", "unknown"),
                    "candidate": self.config.candidate_name,
                    "document_id": document_id,
                    "timestamp": int(time.time()),
                    "processing_version": "1.2"
                }
            )
            for (section_type, content), embedding in zip(sections, embeddings)
        ]
        
        for attempt in range(max_retries):
            try:
                operation_info = self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=points,
                    wait=True,
                    
                )
                print(f"✅ Successfully ingested {len(points)} points")
                return operation_info.status == "completed"
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"⚠️ Retry {attempt+1}/{max_retries} after {delay}s: {str(e)}")
                    time.sleep(delay)
                else:
                    print(f"❌ Error ingesting resume after {max_retries} attempts: {str(e)}")
                    return False

# --------------------------
# Retriever Class
# --------------------------
class ResumeRetriever:
    """Retrieves resume information from vector database"""
    
    def __init__(self, vector_db: ResumeVectorDB, processor: ResumeProcessor):
        self.vector_db = vector_db
        self.processor = processor
        self.score_threshold = 0.7
    
    def query_resume(self, question: str, threshold: float = None, max_retries: int = 3) -> List[Dict]:
        """Search for relevant sections in the resume matching the question"""
        if threshold is not None:
            self.score_threshold = threshold
            
        query_vector = self.processor.embed_text([question])[0]
        
        for attempt in range(max_retries):
            try:
                results = self.vector_db.client.search(
                    collection_name=self.vector_db.config.collection_name,
                    query_vector=query_vector,
                    limit=10,
                    search_params=SearchParams(
                        hnsw_ef=128,
                        exact=False
                    )
                )
                
                filtered_results = []
                for res in results:
                    if res.score >= self.score_threshold:
                        filtered_results.append({
                            "section": res.payload.get("section_type", "Unknown"),
                            "content": res.payload.get("content", ""),
                            "page": res.payload.get("page", 0),
                            "score": res.score,
                            "source": res.payload.get("source", "Unknown")
                        })
                
                print(f"✅ Found {len(filtered_results)} relevant sections")
                return filtered_results
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    print(f"⚠️ Retry {attempt+1}/{max_retries} after {delay}s: {str(e)}")
                    time.sleep(delay)
                else:
                    print(f"❌ Error querying resume after {max_retries} attempts: {str(e)}")
                    return []
    
    def search_by_section(self, section_type: str) -> List[Dict]:
        """Filter results by section type"""
        try:
            points, _ = self.vector_db.client.scroll(
                collection_name=self.vector_db.config.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="section_type",
                            match=models.MatchValue(value=section_type)
                        ),
                    ]
                ),
                limit=100
            )
            
            return [{
                "section": point.payload.get("section_type"),
                "content": point.payload.get("content"),
                "page": point.payload.get("page"),
                "source": point.payload.get("source")
            } for point in points]
            
        except Exception as e:
            print(f"❌ Error searching by section: {str(e)}")
            return []

# --------------------------
# Main RAG System
# --------------------------
class ResumeRAGSystem:
    """Complete RAG system with dynamic candidate name extraction"""
    
    def __init__(self):
        self.config = ResumeConfig()
        self._verify_environment()
        self.processor = ResumeProcessor(self.config)
        self.candidate_name = None
        self.vector_db = None
        self.retriever = None
    
    def _verify_environment(self):
        """Check critical environment variables"""
        if not os.path.exists(".env"):
            print("⚠️ Warning: .env file not found")
        if not self.config.qdrant_url:
            raise RuntimeError("❌ QDRANT_URL must be set in .env file")
    
    def process_resume(self, file_path: str) -> bool:
        """Process and store a resume file with dynamic candidate name extraction"""
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found - {file_path}")
            return False
        
        try:
            with open(file_path, "rb") as f:
                print(f"🔍 Processing resume: {os.path.basename(file_path)}")
                file_bytes = f.read()
                
                self.candidate_name = self.processor.extract_candidate_name(file_bytes)
                self.config.candidate_name = self.candidate_name
                print(f"👤 Identified candidate: {self.candidate_name}")
                
                self.vector_db = ResumeVectorDB(self.config)
                self.retriever = ResumeRetriever(self.vector_db, self.processor)
                
                sections = self.processor.extract_from_pdf(file_bytes)
                if not sections:
                    print("❌ No valid content extracted")
                    return False
                
                texts = [content["text"] for _, content in sections]
                print(f"🧠 Generating embeddings for {len(texts)} chunks...")
                embeddings = self.processor.embed_text(texts)
                
                print("💾 Storing in vector database...")
                success = self.vector_db.ingest_resume(sections, embeddings)
                
                if success:
                    print(f"✅ Successfully processed {len(sections)} sections")
                else:
                    print("❌ Failed to store in database")
                
                return success
        except Exception as e:
            print(f"❌ Error processing resume: {str(e)}")
            return False
    
    def query(self, question: str, threshold: float = 0.7) -> List[Dict]:
        """Query the candidate's resume with enhanced results"""
        if not self.retriever:
            print("❌ Error: You must process a resume before querying")
            return []
            
        try:
            results = self.retriever.query_resume(question, threshold)
            return self._format_results(results)
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            return []
    
    def _format_results(self, raw_results: List[Dict]) -> List[Dict]:
        """Format results for better readability"""
        return [{
            "section": res["section"].upper(),
            "page": res["page"],
            "confidence": f"{res['score']:.1%}",
            "content": res["content"][:500] + ("..." if len(res["content"]) > 500 else ""),
            "source": f"Page {res['page'] + 1}"
        } for res in raw_results]

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    try:
        print("\n🚀 Initializing RAG system...")
        rag = ResumeRAGSystem()
        
        resume_path = "/Users/ShahYash/Desktop/Projects/ask-my-prof/AskMyProf/frontend/rag/YASH.SHAH.RESUME.pdf"
        if rag.process_resume(resume_path):
            print(f"\n✅ Resume processed successfully for {rag.candidate_name}")
            
            queries = [
                "What programming languages are mentioned?",
                "List the candidate's work experience",
                "What education background does the candidate have?",
                "Does the candidate have cloud computing experience?"
            ]
            
            for query in queries:
                print(f"\n🔍 Query: '{query}'")
                results = rag.query(query)
                
                for i, res in enumerate(results, 1):
                    print(f"\n📄 Result {i}:")
                    print(f"Section: {res['section']}")
                    print(f"Source: {res['source']}")
                    print(f"Confidence: {res['confidence']}")
                    print(f"Content:\n{res['content']}\n")
        
        print("\n📊 Database Status:")
        print(f"Collection: {rag.config.collection_name}")
        print(f"Candidate: {rag.config.candidate_name}")
        print(f"Qdrant URL: {rag.config.qdrant_url}")
        
    except Exception as e:
        print(f"❌ System error: {str(e)}")