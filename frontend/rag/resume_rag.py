# --- Imports and Setup (Keep as before) ---
# 9.8/10 best and final code version
import torch
from qdrant_client import models
from qdrant_client import QdrantClient, http # Import http for status codes
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse # Import specific exception
from tqdm import tqdm
import fitz  # PyMuPDF for PDF processing
import os
import glob
from sentence_transformers import SentenceTransformer
import requests
import json
import dotenv
import time
import uuid
from qdrant_client.models import SearchParams
import warnings
import re # Import regex for cleaning
import math # For ceiling division

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
dotenv.load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# --- End Imports and Setup ---

def batch_iterate(lst, batch_size):
    """Iterate over a list in batches."""
    for i in range(0, len(lst), batch_size): yield lst[i : i + batch_size]

def prepare_query(query):
    """Prepare query (placeholder for potential future enhancements)."""
    return query

# --- PDFProcessor Class (MODIFIED for improved section and project handling) ---
class PDFProcessor:
    def __init__(self, model_name='all-mpnet-base-v2'): # Removed chunk_strategy from init as it's handled internally now
        """Initialize with embedding model."""
        self.text_model = SentenceTransformer(model_name)
        self.embedding_dim = self.text_model.get_sentence_embedding_dimension()
        # *** UPDATED Section Headers and Keywords ***
        # Mapping of general section types to specific keywords found in resumes
        self.section_keywords = {
            "EDUCATION": ["EDUCATION", "ACADEMIC BACKGROUND", "ACADEMIC QUALIFICATIONS"],
            "EXPERIENCE": ["EXPERIENCE", "WORK HISTORY", "EMPLOYMENT", "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE"],
            "SKILLS": ["SKILLS", "TECHNICAL SKILLS", "COMPETENCIES", "EXPERTISE"],
            "PROJECTS": ["PROJECTS", "RESEARCH PROJECTS", "PERSONAL PROJECTS", "ACADEMIC PROJECTS", "PAPERS", "RESEARCH", "CONTRIBUTIONS", "PORTFOLIO"], # Expanded project keywords
            "CERTIFICATIONS": ["CERTIFICATIONS", "CERTIFICATES", "ACCREDITATIONS", "COURSES"], # Added COURSES here
            "PUBLICATIONS": ["PUBLICATIONS", "PUBLICATIONS AND PRESENTATIONS"],
            "AWARDS": ["AWARDS", "HONORS", "ACCOMPLISHMENTS"],
            "LANGUAGES": ["LANGUAGES", "LANGUAGE PROFICIENCY"],
            "INTERESTS": ["INTERESTS", "HOBBIES"],
            "SUMMARY": ["SUMMARY", "PROFESSIONAL SUMMARY", "PROFILE", "OBJECTIVE"],
            "CONTACT": ["CONTACT", "CONTACT DETAILS", "PERSONAL DETAILS"]
            # Add other potential sections as needed
        }
        # Flatten keywords for quick lookup during line iteration
        self._all_section_keywords = {keyword: section_type for section_type, keywords in self.section_keywords.items() for keyword in keywords}

        self.chunk_size_limit = 500 # Define a soft chunk size limit for splitting within sections

        print(f"✅ PDFProcessor initialized with model: {model_name} (dim: {self.embedding_dim})")
        print(f"    Recognized Section Types and Keywords: {self.section_keywords}")

    def clean_text(self, text):
        """Clean and normalize text."""
        # Replace various bullet points with a standard one
        text = re.sub(r'[•●○◦◘▪️▫️–*-]', '• ', text)
        # Replace multiple spaces/newlines with a single space (preserving paragraph breaks implicitly)
        text = re.sub(r'\s+', ' ', text).strip()
        # Restore intended newlines for list items or paragraphs if cleaning was too aggressive
        # This regex tries to keep line breaks that look like intentional list items or paragraph separators
        text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text) # Preserve paragraph breaks (2+ newlines)
        text = re.sub(r'\s*\n\s*•', '\n•', text) # Ensure bullet points start on a new line
        text = re.sub(r'•\s+', '• ', text) # Normalize space after bullet point
        return text.strip()


    # *** REVISED extract_from_pdf for section-aware paragraph chunking ***
    def extract_from_pdf(self, file_path):
        """Extract text and chunk it, preserving sections and splitting large sections by paragraph."""
        all_chunks = []
        raw_first_page_text = ""
        current_section = "Introduction"
        current_section_lines = []
        chunk_index_counter = 0

        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                # Use 'text' output for better layout preservation, fallback to plain text
                text = page.get_text("text")
                if not text.strip():
                    text = page.get_text() # Fallback

                if page_num == 0:
                    raw_first_page_text = text # For name extraction

                lines = text.split('\n')
                for line in lines:
                    cleaned_line = line.strip()
                    if not cleaned_line:
                        continue # Skip empty lines

                    is_header = False
                    line_upper = cleaned_line.upper()

                    # Check if the line matches a section header keyword
                    matched_section_type = None
                    for keyword, section_type in self._all_section_keywords.items():
                         # Use word boundaries and check if the keyword is a significant part of the line
                         if re.search(r'\b' + re.escape(keyword) + r'\b', line_upper) and (len(line_upper.split()) <= 5 or line_upper.strip() == keyword):
                             matched_section_type = section_type
                             is_header = True
                             break # Found a header, break from keyword loop

                    if is_header:
                        # Process the previous section's accumulated lines
                        if current_section_lines:
                            # Clean and split the accumulated text by paragraphs
                            section_text = "\n".join(current_section_lines)
                            self._add_section_chunks(all_chunks, current_section, section_text, page_num, chunk_index_counter)
                            chunk_index_counter = len(all_chunks) # Update counter based on added chunks

                        # Start accumulating lines for the new section
                        current_section = matched_section_type
                        current_section_lines = [cleaned_line] # Include the header line in the new section's content
                    else:
                        # Add non-header lines to the current section's lines
                        current_section_lines.append(cleaned_line)

            # Process the last section after the loop finishes
            if current_section_lines:
                 self._add_section_chunks(all_chunks, current_section, "\n".join(current_section_lines), doc.page_count - 1, chunk_index_counter)
                 # No need to update chunk_index_counter here as we are done

            doc.close()
            print(f"📄 Extracted and chunked into {len(all_chunks)} chunks.")

            # Debug: Print first few chunks
            # for i, c in enumerate(all_chunks[:10]):
            #      print(f"  Chunk {i}: Page {c['page']}, Idx: {c['chunk_index']}, Section: {c.get('section', 'N/A')}, Text: {c['text'][:150]}...")

            return all_chunks, raw_first_page_text
        except Exception as e:
            print(f"❌ Error extracting/chunking PDF {os.path.basename(file_path)}: {str(e)}")
            # import traceback
            # traceback.print_exc()
            return [], ""

    def _add_section_chunks(self, all_chunks_list, section_name, section_text, page_num, start_chunk_index):
        """Splits section text into smaller chunks (paragraphs) and adds to the main list."""
        cleaned_section_text = self.clean_text(section_text)
        if not cleaned_section_text:
            return # Don't add empty sections

        # Split the cleaned section text by paragraphs (two or more newlines)
        paragraphs = re.split(r'\n{2,}', cleaned_section_text)
        current_chunk_paragraphs = []
        current_chunk_text = ""
        current_chunk_idx = start_chunk_index

        for para in paragraphs:
            para_cleaned = para.strip()
            if not para_cleaned:
                continue # Skip empty paragraphs

            # Check if adding this paragraph exceeds the chunk size limit
            # Add +1 for the newline that will separate paragraphs in the chunk
            if len(current_chunk_text) + len(para_cleaned) + (1 if current_chunk_text else 0) > self.chunk_size_limit:
                # If we have something in the current chunk, add it
                if current_chunk_text:
                    all_chunks_list.append({
                        "text": f"[Page {page_num + 1}] [{section_name}]\n{current_chunk_text.strip()}",
                        "page": page_num + 1,
                        "chunk_index": current_chunk_idx,
                        "section": section_name
                    })
                    current_chunk_idx += 1
                # Start a new chunk with the current paragraph
                current_chunk_text = para_cleaned + "\n" # Add a newline to separate from next paragraph
            else:
                # Add the paragraph to the current chunk
                if current_chunk_text:
                     current_chunk_text += "\n" # Add newline separator
                current_chunk_text += para_cleaned

        # Add the last accumulated chunk if it's not empty
        if current_chunk_text:
            all_chunks_list.append({
                "text": f"[Page {page_num + 1}] [{section_name}]\n{current_chunk_text.strip()}",
                "page": page_num + 1,
                "chunk_index": current_chunk_idx,
                "section": section_name
            })

    # --- extract_candidate_name (Keep as is) ---
    def extract_candidate_name(self, raw_first_page_text):
        if not raw_first_page_text: return "Unknown Candidate"
        lines = raw_first_page_text.strip().split('\n')[:8]; potential_names = []
        for line in lines:
            cleaned_line = line.strip()
            if not cleaned_line or '@' in cleaned_line or re.search(r'[\d\-]{5,}', cleaned_line) or len(cleaned_line) > 50: continue
            if (cleaned_line.isupper() or cleaned_line.istitle()) and 1 < len(cleaned_line.split()) <= 3:
                 is_likely_header = any(header in cleaned_line.upper() for header in ["EDUCATION", "EXPERIENCE", "SKILLS", "PROJECTS", "SUMMARY", "OBJECTIVE", "CONTACT"])
                 if not is_likely_header and all(re.match(r'^[A-Za-z\s\-]+$', part) for part in cleaned_line.split()): potential_names.append((1, cleaned_line))
        for i, line in enumerate(lines):
            if i > 0:
                prev_line = lines[i-1].strip().lower()
                if prev_line.startswith("name") and ':' in prev_line:
                     cleaned_line = line.strip()
                     if cleaned_line and 1 < len(cleaned_line.split()) <= 4 and len(cleaned_line) < 60: potential_names.append((2, cleaned_line))
        if not potential_names:
             for line in lines:
                 cleaned_line = line.strip()
                 if cleaned_line and '@' not in cleaned_line and not re.search(r'[\d\-()]{5,}', cleaned_line) and len(cleaned_line) < 60:
                     parts = cleaned_line.split()
                     if 1 < len(parts) <= 4 and sum(c.isalpha() for c in cleaned_line) > len(cleaned_line) * 0.6:
                          is_likely_header = any(header in cleaned_line.upper() for header in ["EDUCATION", "EXPERIENCE", "SKILLS", "PROJECTS", "SUMMARY", "OBJECTIVE", "CONTACT"])
                          if not is_likely_header: potential_names.append((3, cleaned_line)); break
        if potential_names:
             potential_names.sort(); best_name = potential_names[0][1]
             # print(f"ℹ️ Extracted potential name (Priority {potential_names[0][0]}): '{best_name}'")
             return best_name
        return "Unknown Candidate"

    # --- embed_text (Keep as is) ---
    def embed_text(self, chunk_dicts, batch_size=32):
        embeddings = []; texts_to_embed = [chunk['text'] for chunk in chunk_dicts if chunk.get('text')]
        if not texts_to_embed: return embeddings
        # print(f"⏳ Generating embeddings for {len(texts_to_embed)} text chunks...")
        # Use math.ceil for ceiling division to get correct batch count
        total_batches = math.ceil(len(texts_to_embed) / batch_size)
        for batch in tqdm(batch_iterate(texts_to_embed, batch_size), total=total_batches, desc="Embedding", leave=False):
            try:
                batch_embeddings = self.text_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                embeddings.extend(batch_embeddings.tolist())
            except Exception as e:
                print(f"❌ Error embedding batch: {e}")
                embeddings.extend([[0.0] * self.embedding_dim] * len(batch))
        # print(f"✅ Embeddings generated.")
        return embeddings

# --- ResumeVectorDB Class (Keep as is) ---
class ResumeVectorDB:
    def __init__(self, collection_name, text_vector_dim, batch_size=128, qdrant_url=None, qdrant_api_key=None): # Increased batch size
        self.collection_name = collection_name; self.batch_size = batch_size; self.text_vector_dim = text_vector_dim
        self.qdrant_url = qdrant_url or QDRANT_URL; self.qdrant_api_key = qdrant_api_key or QDRANT_API_KEY; self.client = None
        if not self.qdrant_url or not self.qdrant_api_key: raise ValueError("Qdrant URL and API Key must be provided.")

    def define_client(self):
        try:
            self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=180)
            self.client.get_collections(); # print(f"✅ Qdrant client initialized: {self.qdrant_url}")
        except Exception as e: print(f"❌ Failed to initialize Qdrant client: {e}"); self.client = None

    def create_collection(self, force_recreate=False):
        if not self.client: print("❌ Cannot create collection, client not initialized."); return
        text_collection_name = f"{self.collection_name}_text"
        try:
            collection_exists = False
            try: self.client.get_collection(collection_name=text_collection_name); collection_exists = True; # print(f"ℹ️ Collection '{text_collection_name}' exists.")
            except UnexpectedResponse as e:
                 if hasattr(e, 'status_code') and e.status_code == 404: collection_exists = False # print(f"ℹ️ Collection '{text_collection_name}' not found.")
                 else: raise e
            except Exception as e: print(f"⚠️ Error checking collection: {e}"); collection_exists = False
            if force_recreate and collection_exists:
                print(f"⏳ Deleting existing collection '{text_collection_name}'..."); self.client.delete_collection(collection_name=text_collection_name, timeout=60)
                print(f"✅ Collection '{text_collection_name}' deleted."); collection_exists = False; time.sleep(2)
            if not collection_exists:
                print(f"⏳ Creating collection: {text_collection_name}..."); self.client.create_collection(collection_name=text_collection_name, vectors_config=VectorParams(size=self.text_vector_dim, distance=Distance.COSINE))
                print(f"✅ Created collection: {text_collection_name}")
        except Exception as e: print(f"❌ Error managing collection '{text_collection_name}': {e}")

    def ingest_text_data(self, text_embeddings, chunk_dicts, document_id, candidate_id, candidate_name):
        if not self.client: print("❌ Cannot ingest, client not initialized."); return False
        text_collection_name = f"{self.collection_name}_text"
        try:
            points = []
            if len(text_embeddings) != len(chunk_dicts):
                 print(f"❌ Error: Mismatch embed ({len(text_embeddings)}) vs chunk ({len(chunk_dicts)}) for {document_id}. Skip."); return False
            for i, embedding in enumerate(text_embeddings):
                chunk_data = chunk_dicts[i]
                points.append(models.PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={
                            "text": chunk_data.get('text', ''), "page": chunk_data.get('page', -1),
                            "chunk_index": chunk_data.get('chunk_index', -1), "section": chunk_data.get('section', 'Unknown'),
                            "document_id": document_id, "candidate_id": candidate_id, "candidate_name": candidate_name,
                        }))
            if not points: print(f"ℹ️ No points to ingest for '{document_id}'."); return True
            # print(f"⏳ Upserting {len(points)} points for doc '{document_id}'...")
            # Upsert in batches using the batch_iterate helper
            total_batches = math.ceil(len(points) / self.batch_size)
            for batch in tqdm(batch_iterate(points, self.batch_size), total=total_batches, desc=f"Upserting {document_id}", leave=False):
                 operation_info = self.client.upsert(collection_name=text_collection_name, points=batch, wait=True)
                 if operation_info.status != models.UpdateStatus.COMPLETED:
                      print(f"⚠️ Batch upsert issue for doc '{document_id}', status: {operation_info.status}. Check logs.")
                      # Optionally return False or raise an error here
            # print(f"✅ Upsert completed for doc '{document_id}'.")
            return True
        except Exception as e: print(f"❌ Error during upsert for doc '{document_id}': {e}"); return False


# --- ResumeRetriever Class (Keep as is) ---
class ResumeRetriever:
    def __init__(self, vector_db, pdf_processor):
        self.vector_db = vector_db
        self.pdf_processor = pdf_processor
        # *** UPDATED Query Expansions (Keep the improved list) ***
        self.query_expansions = {
            "skills": ["technologies", "programming", "tools", "frameworks", "libraries", "technical skills", "competencies"],
            "experience": ["work", "job", "employment", "professional experience", "role", "position"],
            "education": ["degree", "university", "college", "academic", "qualification"],
            "projects": ["portfolio", "personal projects", "academic projects", "papers", "research", "development", "contributions"], # Added relevant terms
            "contact": ["email", "phone", "address", "contact details", "location"],
            "courses": ["certifications", "training", "online courses", "professional development", "credential"],
            "internship": ["intern", "trainee", "co-op", "work placement"],
            "publications": ["papers", "research papers", "articles"] # Added publications
        }
        # print(f"ℹ️ ResumeRetriever initialized.")

    def _simple_expand_query(self, query):
        """Expands query using simple keyword matching and synonyms."""
        expanded_terms = [query.strip()] # Start with the original query cleaned
        query_lower = query.lower()
        matched = False
        for keyword, expansions in self.query_expansions.items():
            # Use word boundaries for keyword matching to avoid partial matches
            if re.search(rf"\b{keyword}\b", query_lower):
                # Add expansions BUT prioritize the original query structure
                # Simple approach: just append relevant terms
                expanded_terms.extend([exp for exp in expansions if exp not in query_lower])
                matched = True

        # If no keyword matched, maybe just use the original query
        if not matched and len(query.split()) > 5: # Don't expand long, specific queries if no keyword hit
             return query.strip()

        # Create unique terms, keep order roughly, join
        # Using dict.fromkeys preserves order in Python 3.7+
        unique_terms = list(dict.fromkeys(expanded_terms))
        expanded_query = " ".join(unique_terms)

        # Only log if expansion actually changed the query meaningfully
        if expanded_query.lower() != query.lower().strip():
             # print(f"    🔍 Expanded query: '{query}' -> '{expanded_query}'")
             return expanded_query
        return query.strip() # Return original cleaned query if no change


    def search(self, query, candidate_id=None, top_k=50, max_retries=3, initial_delay=1): # Increased top_k
        """Search Qdrant with query expansion, retry logic."""
        if not self.vector_db.client: print("❌ Cannot search, client not initialized."); return []

        original_query = query
        expanded_query = self._simple_expand_query(original_query)
        prepared_query = prepare_query(expanded_query)

        try: query_vector = self.pdf_processor.text_model.encode(prepared_query).tolist()
        except Exception as e: print(f"❌ Error encoding query '{prepared_query}': {e}"); return []

        text_collection_name = f"{self.vector_db.collection_name}_text"
        search_filter = models.Filter(must=[models.FieldCondition(key="candidate_id", match=models.MatchValue(value=candidate_id))]) if candidate_id else None
        # if candidate_id: print(f"ℹ️ Searching within cand_id: {candidate_id}")

        for attempt in range(max_retries):
            try:
                search_result = self.vector_db.client.search(
                    collection_name=text_collection_name, query_vector=query_vector, query_filter=search_filter,
                    limit=top_k, search_params=SearchParams(hnsw_ef=128, exact=False), # hnsw_ef=128 good balance
                    with_payload=True, score_threshold=0.0 # Retrieve all, filter in RAG
                )
                # print(f"✅ [DEBUG] Retrieved {len(search_result)} raw chunks from Qdrant for query: '{prepared_query}'.")
                # if search_result:
                #     scores = [p.score for p in search_result]
                #     print(f"    Scores range: [{min(scores):.4f} - {max(scores):.4f}]")
                return search_result
            except UnexpectedResponse as e: print(f"⚠️ Qdrant search error (Attempt {attempt+1}/{max_retries}, HTTP {getattr(e, 'status_code', 'N/A')}): {e}")
            except Exception as e: print(f"❌ Unexpected search error (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt >= max_retries - 1: print("❌ Max search retries reached."); return []
            wait_time = initial_delay * (2 ** attempt); # print(f"Retrying search in {wait_time}s...")
            time.sleep(wait_time)
        return []


# --- ResumeRAG Class (Keep as is) ---
class ResumeRAG:
    def __init__(self, retriever, model_name="gemma2-9b-it", api_key=None):
        self.model_name = model_name; self.api_key = api_key or GROQ_API_KEY
        if not self.api_key: raise ValueError("Groq API key required")
        self.retriever = retriever; 
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        # RAG Parameters
        self.base_threshold = 0.15; self.min_threshold = 0.10
        self.context_limit = 7000 # Slightly increased limit
        self.max_chunks_in_context = 25 # Increased max chunks limit
        # Keywords indicating list-type questions for context prioritization
        self.list_query_keywords = ["list", "what are", "skills", "projects", "courses", "certifications", "experience", "languages", "tools", "academic projects", "personal projects", "research projects", "papers", "publications"] # Added project keywords
        # print(f"✅ ResumeRAG initialized with model: {self.model_name}, Context Limit: {self.context_limit} chars, Max Chunks: {self.max_chunks_in_context}")

    # Helper to identify query type
    def _is_list_query(self, query):
        return any(keyword in query.lower() for keyword in self.list_query_keywords)

    # *** REVISED Context Generation with Section Prioritization (Keep the improved logic) ***
    def generate_context(self, query, candidate_id=None):
        """Generate context, prioritizing chunks from sections relevant to the query type."""
        text_results = self.retriever.search(query, candidate_id=candidate_id, top_k=50) # Retrieve more initially
        if not text_results: return "No relevant information found in the resume(s)."

        sorted_results = sorted(text_results, key=lambda x: x.score, reverse=True)
        max_score = sorted_results[0].score if sorted_results else 0

        # Dynamic threshold
        rag_threshold = self.base_threshold
        # Simple dynamic adjustment based on max score
        if max_score < 0.4:
            rag_threshold = max(self.min_threshold, max_score * 0.5) # Be more lenient if top score is low
        elif max_score > 0.7:
            rag_threshold = max(self.base_threshold, max_score * 0.2) # Be stricter if top score is very high


        # Filter by threshold
        threshold_filtered_results = [point for point in sorted_results if point.score >= rag_threshold]

        if not threshold_filtered_results:
            # print(f"    ⚠️ No chunks above RAG threshold ({rag_threshold:.4f}). Falling back to top few.")
            # Fallback to top N chunks regardless of threshold if none pass
            threshold_filtered_results = sorted_results[:8] # Fallback to top 8 chunks
            if not threshold_filtered_results: return "No relevant information found in the resume(s)."


        # --- Context Selection with Prioritization ---
        final_context_chunks = []
        current_context_length = 0
        seen_texts = set()
        used_point_ids = set() # Track IDs of points already added


        # If it's a list-type query, try to ensure relevant sections are represented
        if self._is_list_query(query):
            # Identify potentially relevant sections based on query keywords
            relevant_section_keywords = []
            query_lower = query.lower()
            # Increased keywords for projects and publications
            if any(k in query_lower for k in ["project", "paper", "research", "publication", "contributions", "academic projects", "personal projects", "research projects", "papers"]):
                 relevant_section_keywords.extend(["PROJECTS", "PAPERS", "ACADEMIC", "PUBLICATIONS", "RESEARCH"]) # Ensure these match section names in payload
            if any(k in query_lower for k in ["skill", "technolog", "tool", "programming", "frameworks", "libraries", "competencies"]): relevant_section_keywords.extend(["SKILLS", "TECHNICAL SKILLS"])
            if any(k in query_lower for k in ["course", "certificat", "training", "credential"]): relevant_section_keywords.extend(["COURSES", "CERTIFICATIONS"])
            if any(k in query_lower for k in ["experience", "job", "intern", "work"]): relevant_section_keywords.extend(["EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE"])
            if any(k in query_lower for k in ["education", "degree", "academic background"]): relevant_section_keywords.extend(["EDUCATION"])
            if any(k in query_lower for k in ["contact", "email", "phone", "address"]): relevant_section_keywords.extend(["CONTACT", "PERSONAL DETAILS"])


            # Select top chunk from each relevant section found in filtered results
            prioritized_sections_added = set()
            for point in threshold_filtered_results:
                section = point.payload.get("section", "Unknown").upper()
                point_id = point.id

                # Check if this section matches keywords and hasn't had its top chunk added yet
                # Also check if the section name in the payload contains any of the relevant keywords
                if any(key in section for key in relevant_section_keywords) and section not in prioritized_sections_added:
                     entry_text = point.payload.get("text", "")
                     prefix = f"[Source: ... Relevance={point.score:.3f}]\n"
                     # Check length before adding
                     if current_context_length + len(prefix) + len(entry_text) <= self.context_limit:
                         if entry_text not in seen_texts:
                             final_context_chunks.append(point) # Add the actual point
                             seen_texts.add(entry_text)
                             current_context_length += len(prefix) + len(entry_text)
                             used_point_ids.add(point_id)
                             prioritized_sections_added.add(section) # Mark section as covered for prioritization
                             if len(final_context_chunks) >= self.max_chunks_in_context: break # Stop if max chunks hit


        # Fill remaining context space with highest-scoring chunks not already included
        remaining_slots = self.max_chunks_in_context - len(final_context_chunks)
        if remaining_slots > 0:
             for point in threshold_filtered_results:
                 point_id = point.id
                 if point_id not in used_point_ids:
                     entry_text = point.payload.get("text", "")
                     prefix = f"[Source: ... Relevance={point.score:.3f}]\n"
                     if current_context_length + len(prefix) + len(entry_text) <= self.context_limit:
                         if entry_text not in seen_texts:
                              final_context_chunks.append(point)
                              seen_texts.add(entry_text)
                              current_context_length += len(prefix) + len(entry_text)
                              used_point_ids.add(point_id)
                              remaining_slots -= 1
                              if remaining_slots <= 0: break # Stop if slots filled or context limit hit
                     else: # Stop if next chunk exceeds limit
                          # print(f"    ⚠️ Context length limit ({self.context_limit} chars) reached during fill.")
                          break

        if not final_context_chunks: return "No relevant context could be selected."

        # Sort final selection by score for better presentation to LLM? Optional.
        final_context_chunks.sort(key=lambda x: x.score, reverse=True)

        # Format the final context string
        context_entries = []
        for point in final_context_chunks:
             prefix = f"[Source: Doc='{point.payload.get('document_id', 'N/A')}', Cand='{point.payload.get('candidate_name', 'N/A')}', Page={point.payload.get('page', 'N/A')}, Section='{point.payload.get('section', 'N/A')}', Relevance={point.score:.3f}]\n"
             context_entries.append(prefix + point.payload.get("text", ""))

        # print(f"ℹ️ RAG Context: Using {len(final_context_chunks)} prioritized/selected chunks (Score >= {rag_threshold:.4f}, Max Score: {max_score:.4f}).")
        text_context = "\n\n---\n\n".join(context_entries)
        return f"Context from Resume(s) (Score Threshold: {rag_threshold:.3f}, Max Score: {max_score:.3f}, Chunks: {len(final_context_chunks)}):\n---------------------\n{text_context}\n---------------------"


    # *** REVISED Query Prompt (Keep the improved prompt) ***
    def query(self, query, candidate_id=None, candidate_name=None):
        """Query the LLM with revised context generation and prompt."""
        # print(f"\n⏳ Generating context for query: '{query}'" + (f" for cand_id: '{candidate_id}'" if candidate_id else ""))
        context = self.generate_context(query=query, candidate_id=candidate_id)

        if not context or context.strip().startswith("No relevant information found"):
            print("❌ No relevant context generated.")
            no_context_prompt = f"""User question: "{query}"
            Candidate: ID='{candidate_id}', Name='{candidate_name}'.
            Instructions: Inform the user clearly and concisely that the requested information ('{query}') could not be found in the provided resume excerpts based on relevance matching. Do not invent information."""
            return self.call_llm(no_context_prompt, is_no_context=True)

        # print(f"✅ Context generated. Querying LLM: {self.model_name}")
        candidate_info = f"Candidate Info: ID='{candidate_id}', Name='{candidate_name}'\n\n" if candidate_id else ""

        # --- REVISED PROMPT TEMPLATE v5 (Emphasis on Project Extraction) ---
        qa_prompt_tmpl_str = f"""**Resume Analysis Task**

        **User Question:** "{query}"

        {candidate_info}
        **Resume Information (Context - Extracted relevant excerpts):**
        {context}

        **Instructions for Answering:**
        1.  **Strictly Adhere to Context:** Base your answer ONLY on the information explicitly present in the 'Resume Information (Context)' provided above. Do NOT infer or add information not present.
        2.  **Address the Question Directly:** Answer the user's question precisely.
        3.  **Extract and List Completely (Especially for Projects/Papers):** If the question asks for a list (e.g., "list skills", "what projects", "courses", "certifications", "papers"), extract ALL relevant items mentioned within the context. **PAY CLOSE ATTENTION TO PROJECT/PAPER DETAILS.** Look for chunks tagged with 'PROJECTS', 'ACADEMIC PROJECTS', 'PERSONAL PROJECTS', 'RESEARCH PROJECTS', 'PAPERS', 'PUBLICATIONS', or similar (check the Section tag in the source info). For each project/paper identified, extract its name, description, and specifically list the technologies used if mentioned in the context. Synthesize information if items are split across context chunks from the *same logical section* (like parts of the same project description). Present lists clearly using bullet points.
        4.  **Acknowledge Missing Information:**
            * If the context contains the relevant section(s) (e.g., 'ACADEMIC PROJECTS AND PAPERS' is present) but the *specific detail* requested isn't mentioned (e.g., asking about a project not listed), state that the detail isn't mentioned in the provided excerpts.
            * If the *entire relevant section* seems absent from the provided context excerpts (e.g., no chunks tagged with 'PROJECTS' or similar were included in the context), state "The provided resume excerpts do not seem to contain details about [Requested Topic, e.g., Projects, Papers]."
        5.  **Be Concise:** Provide the information directly.

        **Answer:**"""
        # --- END REVISED PROMPT TEMPLATE v5 ---

        return self.call_llm(qa_prompt_tmpl_str)

    # --- call_llm (Keep as is) ---
    def call_llm(self, prompt, is_no_context=False):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        system_message = "You are an expert resume analysis assistant. Answer questions based *strictly* on the provided resume excerpts (context). Extract information accurately and completely. If information is missing from the context, state that clearly."
        if is_no_context: system_message = "You are a helpful assistant. Inform the user concisely that the requested information was not found in the relevant resume excerpts provided."
        payload = {"model": self.model_name, "messages": [{"role": "system", "content": system_message},{"role": "user", "content": prompt}],
                   "temperature": 0.0, "max_tokens": 2000, "top_p": 0.9 }
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=180)
            response.raise_for_status()
            result = response.json()
            if result.get("choices") and result["choices"][0].get("message") and result["choices"][0]["message"].get("content"):
                # print("✅ LLM response received.")
                llm_content = result["choices"][0]["message"]["content"].strip()
                finish_reason = result["choices"][0].get("finish_reason", "N/A")
                # if finish_reason != "stop": print(f"    ⚠️ LLM finish reason: {finish_reason}")
                return llm_content
            elif "error" in result: print(f"❌ Groq API error: {result['error'].get('message', 'Unknown')}"); return f"Error: AI API error - {result['error'].get('message', 'Unknown')}"
            else: print(f"❌ Unexpected Groq API response: {result}"); return "Error: Unexpected AI API response."
        except requests.exceptions.Timeout: print(f"❌ Groq API timed out."); return "Error: AI API request timed out."
        except requests.exceptions.HTTPError as http_err:
            error_detail = http_err.response.text; print(f"❌ Groq API HTTP error: {http_err.response.status_code} - {error_detail}"); return f"Error: AI API request failed ({http_err.response.status_code})."
        except requests.exceptions.RequestException as req_err: print(f"❌ Groq API request error: {req_err}"); return f"Error: AI API communication failed. {req_err}"
        except Exception as e: print(f"❌ Unexpected error during LLM query: {e}"); import traceback; traceback.print_exc(); return f"Error: Unexpected error. {e}"


# --- Main Execution Logic (Keep as is) ---
def process_resumes(directory_path, collection_name="resumes_db", force_recreate_collection=False):
    pdf_paths = glob.glob(os.path.join(directory_path, "*.pdf"))
    if not pdf_paths: print(f"❌ No PDF files found: {directory_path}"); return None, None
    print(f"Found {len(pdf_paths)} PDF files.")

    # Initialize PDFProcessor without chunk_strategy as it's handled internally
    pdf_processor = PDFProcessor(model_name='all-mpnet-base-v2')
    vector_db = ResumeVectorDB(collection_name=collection_name, text_vector_dim=pdf_processor.embedding_dim)
    vector_db.define_client();
    if not vector_db.client: print("❌ Halting: Qdrant client failed."); return None, None
    vector_db.create_collection(force_recreate=force_recreate_collection)

    processed_count = 0; ingested_count = 0; candidate_map = {}
    total_start_time = time.time()

    for pdf_path in pdf_paths:
        file_start_time = time.time()
        filename = os.path.basename(pdf_path)
        document_id = os.path.splitext(filename)[0].upper()
        print(f"\n--- Processing: {filename} ---")
        processed_count += 1

        chunks, raw_first_page = pdf_processor.extract_from_pdf(pdf_path)
        if not chunks: print(f"⚠️ No chunks extracted from {filename}, skipping."); continue
        # print(f"    Extracted {len(chunks)} chunks.")

        candidate_name = pdf_processor.extract_candidate_name(raw_first_page)
        candidate_id = re.sub(r'\s+', '_', candidate_name.lower().strip()) if candidate_name != "Unknown Candidate" else document_id.lower() + "_candidate"
        # print(f"    Cand Name: '{candidate_name}', Cand ID: '{candidate_id}'")

        if candidate_id not in candidate_map: candidate_map[candidate_id] = {"name": candidate_name, "docs": []}
        if document_id not in candidate_map[candidate_id]["docs"]: candidate_map[candidate_id]["docs"].append(document_id)
        if candidate_name != "Unknown Candidate": candidate_map[candidate_id]["name"] = candidate_name

        text_embeddings = pdf_processor.embed_text(chunks)
        if not text_embeddings or len(text_embeddings) != len(chunks):
            print(f"⚠️ Embedding failed/mismatch for {filename}, skipping ingest."); continue

        success = vector_db.ingest_text_data(text_embeddings, chunks, document_id, candidate_id, candidate_name)
        if success: ingested_count += 1; print(f"    Ingested {len(chunks)} points for {document_id} ({time.time() - file_start_time:.2f}s)")
        else: print(f"⚠️ Failed to ingest data for {filename}.")

    print(f"\n--- Processing Summary ---")
    print(f"Total PDFs Processed: {processed_count}")
    print(f"Documents Ingested/Updated: {ingested_count}")
    print(f"Total Unique Candidates: {len(candidate_map)}")
    print(f"Total Processing & Ingestion Time: {time.time() - total_start_time:.2f} seconds")


    retriever = ResumeRetriever(vector_db, pdf_processor)
    rag = ResumeRAG(retriever)
    return rag, candidate_map

# --- Main execution block ---
if __name__ == "__main__":
    main_start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resumes_directory = "/Users/ShahYash/Desktop/Projects/ask-my-prof/AskMyProf/frontend/rag/test_resumes" # ADJUST PATH IF NEEDED
    qdrant_collection = "candidate_resumes_v6" # New version for improved chunking/prompting
    force_recreate = True # Set to True to re-ingest with new chunking
    print(f"--- Configuration ---")
    print(f"Resumes Directory: {resumes_directory}")
    print(f"Qdrant Collection: {qdrant_collection}")
    print(f"Force Recreate: {force_recreate}")
    print(f"--------------------")

    if not os.path.isdir(resumes_directory): print(f"❌ Error: Directory not found: {resumes_directory}")
    else:
        print("\n🚀 Starting Resume Processing & RAG Setup...")
        resume_rag, candidates = process_resumes(resumes_directory, collection_name=qdrant_collection, force_recreate_collection=force_recreate)

        if resume_rag and candidates:
            print("\n✅ RAG System Ready.")
            target_candidate_id = None; target_candidate_name = "Yash Shah"
            for cid, cinfo in candidates.items():
                if cinfo["name"].lower() == target_candidate_name.lower(): target_candidate_id = cid; break

            if target_candidate_id:
                 print(f"\n🎯 Querying for candidate: '{target_candidate_name}' (ID: '{target_candidate_id}')")
                 queries_to_run = [
                     "What is the candidate's name and contact details?",
                     "List all technical skills mentioned.",
                     "Where did Yash Shah intern as an AI Software Engineer and what did he do there?",
                     # *** TESTING PROJECT QUERY AGAIN ***
                     "List all the academic projects and papers Yash Shah worked on, including technologies used for each.",
                     "What courses and certifications are listed?",
                     "Summarize the candidate's education.",
                     "What programming languages does the candidate know?"
                 ]
                 for query in queries_to_run:
                     print(f"\n❓ Query: {query}")
                     q_start_time = time.time()
                     result = resume_rag.query(query=query, candidate_id=target_candidate_id, candidate_name=target_candidate_name)
                     print(f"💬 Answer:\n{result}")
                     print(f"   (Query time: {time.time() - q_start_time:.2f}s)")
                     print("-" * 30)
            else: print(f"\n⚠️ Cannot run queries: Candidate '{target_candidate_name}' not found.")
        else: print("❌ Error: Resume RAG system could not be initialized.")

    print(f"\n--- Total Execution Time: {time.time() - main_start_time:.2f} seconds ---")