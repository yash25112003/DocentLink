# --- Imports and Setup (Keep as before) ---
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
dotenv.load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# --- End Imports and Setup ---

def batch_iterate(lst, batch_size):
    for i in range(0, len(lst), batch_size): yield lst[i : i + batch_size]

def prepare_query(query): return query

# --- PDFProcessor Class (MODIFIED) ---
class PDFProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_strategy='paragraph'):
        self.text_model = SentenceTransformer(model_name)
        self.embedding_dim = self.text_model.get_sentence_embedding_dimension()
        self.chunk_strategy = chunk_strategy
        print(f"✅ PDFProcessor initialized with model: {model_name} (dim: {self.embedding_dim}), Chunk Strategy: {self.chunk_strategy}")

    def chunk_text_by_paragraph(self, text, page_num):
        """Splits text into paragraphs, adding page context."""
        # Split by double newline, then filter empty strings after stripping
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        for i, para in enumerate(paragraphs):
            # Simple heuristic: ignore very short paragraphs (likely list items handled separately or headers)
            if len(para) > 50: # Adjust length threshold as needed
                chunks.append({
                    "text": f"[Page {page_num + 1}] {para}",
                    "page": page_num + 1,
                    "chunk_index": i # Index within the page
                 })
        return chunks

    def extract_from_pdf(self, file_path):
        """Extract text and chunk it based on the chosen strategy."""
        all_chunks = []
        raw_first_page_text = ""
        try:
            doc = fitz.open(file_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if page_num == 0:
                    raw_first_page_text = text # For name extraction

                if text.strip():
                    if self.chunk_strategy == 'paragraph':
                        page_chunks = self.chunk_text_by_paragraph(text, page_num)
                        all_chunks.extend(page_chunks)
                    else: # Default to page chunking (as before)
                         all_chunks.append({
                            "text": f"[Page {page_num + 1}] {text}",
                            "page": page_num + 1,
                            "chunk_index": 0
                         })
            doc.close()
            print(f"📄 Extracted and chunked into {len(all_chunks)} chunks using '{self.chunk_strategy}' strategy.")
            return all_chunks, raw_first_page_text
        except Exception as e:
            print(f"❌ Error extracting/chunking PDF {os.path.basename(file_path)}: {str(e)}")
            return [], ""

    # --- extract_candidate_name and embed_text methods (Keep as in previous version) ---
    def extract_candidate_name(self, raw_first_page_text):
        """Extracts a candidate name from the beginning of the raw resume text."""
        if not raw_first_page_text: return "Unknown Candidate"
        lines = raw_first_page_text.strip().split('\n')
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line and '@' not in cleaned_line and not any(c.isdigit() for c in cleaned_line) and sum(c.isalpha() for c in cleaned_line) > len(cleaned_line) / 2 :
                potential_name = cleaned_line[:60]
                if ' ' in potential_name.strip():
                    print(f"ℹ️ Extracted potential name: '{potential_name.strip()}'")
                    return potential_name.strip()
        return "Unknown Candidate"

    def embed_text(self, chunk_dicts, batch_size=32):
        """Generate embeddings for text content within chunk dictionaries."""
        embeddings = []
        texts_to_embed = [chunk['text'] for chunk in chunk_dicts if chunk.get('text')]
        if not texts_to_embed: return embeddings

        print(f"⏳ Generating embeddings for {len(texts_to_embed)} text chunks...")
        for batch in tqdm(batch_iterate(texts_to_embed, batch_size), total= -(-len(texts_to_embed) // batch_size), desc="Embedding"):
            try:
                batch_embeddings = self.text_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                embeddings.extend(batch_embeddings.tolist())
            except Exception as e:
                print(f"❌ Error embedding batch: {e}")
                embeddings.extend([[0.0] * self.embedding_dim] * len(batch))
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"✅ Embeddings generated.")
        return embeddings


# --- ResumeVectorDB Class (MODIFIED for new chunk structure) ---
class ResumeVectorDB:
    # --- __init__, define_client, create_collection, check_document_exists (Keep as in previous version) ---
    def __init__(self, collection_name, text_vector_dim, batch_size=64, qdrant_url=None, qdrant_api_key=None):
        self.collection_name = collection_name; self.batch_size = batch_size; self.text_vector_dim = text_vector_dim
        self.qdrant_url = qdrant_url or QDRANT_URL; self.qdrant_api_key = qdrant_api_key or QDRANT_API_KEY; self.client = None
        if not self.qdrant_url or not self.qdrant_api_key: raise ValueError("Qdrant URL and API Key must be provided.")

    def define_client(self):
        try:
            self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=120)
            self.client.get_collections(); print(f"✅ Qdrant client initialized: {self.qdrant_url}")
        except Exception as e: print(f"❌ Failed to initialize Qdrant client: {e}"); self.client = None

    def create_collection(self, force_recreate=False):
        if not self.client: print("❌ Cannot create collection, client not initialized."); return
        text_collection_name = f"{self.collection_name}_text"
        try:
            collection_exists = False
            try: self.client.get_collection(collection_name=text_collection_name); collection_exists = True; print(f"ℹ️ Collection '{text_collection_name}' exists.")
            except UnexpectedResponse as e:
                 if e.status_code == 404: print(f"ℹ️ Collection '{text_collection_name}' not found."); collection_exists = False
                 else: raise e
            except Exception as e: print(f"⚠️ Error checking collection: {e}"); collection_exists = False
            if force_recreate and collection_exists:
                print(f"⏳ Deleting existing collection '{text_collection_name}'..."); self.client.delete_collection(collection_name=text_collection_name)
                print(f"✅ Collection '{text_collection_name}' deleted."); collection_exists = False
            if not collection_exists:
                print(f"⏳ Creating collection: {text_collection_name}..."); self.client.create_collection(collection_name=text_collection_name, vectors_config=VectorParams(size=self.text_vector_dim, distance=Distance.COSINE))
                print(f"✅ Created collection: {text_collection_name}")
        except Exception as e: print(f"❌ Error managing collection '{text_collection_name}': {e}")

    def check_document_exists(self, document_id, candidate_id):
        if not self.client: return False
        text_collection_name = f"{self.collection_name}_text"
        try:
            response, _ = self.client.scroll(collection_name=text_collection_name, scroll_filter=models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)), models.FieldCondition(key="candidate_id", match=models.MatchValue(value=candidate_id))]), limit=1, with_payload=False, with_vectors=False)
            if response: print(f"ℹ️ Check found existing data for doc '{document_id}', cand '{candidate_id}'."); return True
            return False
        except UnexpectedResponse as e:
             if e.status_code == 404: return False
             print(f"❌ Error checking doc existence (HTTP {e.status_code}): {e}"); return False
        except Exception as e: print(f"❌ Error checking doc existence: {e}"); return False

    # MODIFIED ingest_text_data
    def ingest_text_data(self, text_embeddings, chunk_dicts, document_id, candidate_id, candidate_name):
        """Ingest resume chunk data into Qdrant."""
        if not self.client: print("❌ Cannot ingest, client not initialized."); return False
        text_collection_name = f"{self.collection_name}_text"
        # self.check_document_exists(document_id, candidate_id) # Optional check

        try:
            points = []
            # Ensure embeddings and chunk_dicts align
            if len(text_embeddings) != len(chunk_dicts):
                 print(f"❌ Error: Mismatch between embeddings ({len(text_embeddings)}) and chunks ({len(chunk_dicts)}) for doc {document_id}. Skipping ingestion.")
                 return False

            for i, embedding in enumerate(text_embeddings):
                chunk_data = chunk_dicts[i]
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            # Ensure all keys exist in chunk_data, provide defaults if necessary
                            "text": chunk_data.get('text', ''), # Get text from dict
                            "page": chunk_data.get('page', -1), # Get page num
                            "chunk_index": chunk_data.get('chunk_index', -1), # Get chunk index
                            "document_id": document_id,
                            "candidate_id": candidate_id,
                            "candidate_name": candidate_name,
                        }
                    )
                )
            if not points: print(f"ℹ️ No points to ingest for '{document_id}'."); return False

            print(f"⏳ Upserting {len(points)} points for doc '{document_id}' (Cand: '{candidate_name}') in batches...")
            for batch in tqdm(batch_iterate(points, self.batch_size), total=-(-len(points)//self.batch_size), desc="Upserting"):
                self.client.upsert(collection_name=text_collection_name, points=batch, wait=True)
            print(f"✅ Successfully upserted doc '{document_id}' for candidate '{candidate_name}'.")
            return True
        except Exception as e: print(f"❌ Error inserting doc '{document_id}': {e}"); return False


# --- ResumeRetriever Class (Keep as in previous version - debug mode prints raw results) ---
class ResumeRetriever:
    def __init__(self, vector_db, pdf_processor):
        self.vector_db = vector_db; self.pdf_processor = pdf_processor
        print(f"ℹ️ ResumeRetriever initialized [DEBUG MODE: No score threshold filtering]")

    def search(self, query, candidate_id=None, top_k=15, max_retries=3, initial_delay=1): # Increased top_k again
        if not self.vector_db.client: print("❌ Cannot search, client not initialized."); return []
        prepared_query = prepare_query(query)
        try: query_vector = self.pdf_processor.text_model.encode(prepared_query).tolist()
        except Exception as e: print(f"❌ Error encoding query '{query}': {e}"); return []
        text_collection_name = f"{self.vector_db.collection_name}_text"
        search_filter = None
        if candidate_id: search_filter = models.Filter(must=[models.FieldCondition(key="candidate_id", match=models.MatchValue(value=candidate_id))]); print(f"ℹ️ Searching within cand_id: {candidate_id}")
        else: print(f"ℹ️ Searching across all resumes.")

        for attempt in range(max_retries):
            try:
                search_result = self.vector_db.client.search(collection_name=text_collection_name, query_vector=query_vector, query_filter=search_filter, limit=top_k, search_params=SearchParams(hnsw_ef=128, exact=False), with_payload=True)
                print(f"✅ [DEBUG] Retrieved {len(search_result)} raw chunks from Qdrant.")
                if search_result:
                    print("--- Retrieved Chunks (Top {}): ---".format(len(search_result)))
                    for i, point in enumerate(search_result):
                        text_snippet = point.payload.get('text', 'N/A')[:150].replace('\n', ' ') + "..."; page_num = point.payload.get('page', 'N/A'); chunk_idx = point.payload.get('chunk_index', 'N/A')
                        print(f"  {i+1}. Score: {point.score:.4f} | Doc: {point.payload.get('document_id', 'N/A')} | Cand: {point.payload.get('candidate_id', 'N/A')} | Page: {page_num} | Chunk: {chunk_idx}")
                        print(f"     Text: {text_snippet}")
                    print("---------------------------------")
                return search_result # Return all results for debugging RAG step
            except UnexpectedResponse as e: print(f"❌ Qdrant search error (Attempt {attempt + 1}/{max_retries}, HTTP {e.status_code}): {e}");
            except Exception as e: print(f"❌ Qdrant search error (Attempt {attempt + 1}/{max_retries}): {e}");
            if attempt >= max_retries - 1: print("❌ Max retries reached."); return []
            wait_time = initial_delay * (2 ** attempt); print(f"Retrying in {wait_time}s..."); time.sleep(wait_time)
        return []


# --- ResumeRAG Class (Keep as in previous version - applies RAG threshold) ---
class ResumeRAG:
    def __init__(self, retriever, model_name="gemma2-9b-it", api_key=None):
        self.model_name = model_name; self.api_key = api_key or GROQ_API_KEY
        if not self.api_key: raise ValueError("Groq API key required")
        self.retriever = retriever; self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        print(f"✅ ResumeRAG initialized with model: {self.model_name}")

    def generate_context(self, query, candidate_id=None):
        text_results = self.retriever.search(query, candidate_id=candidate_id) # Gets all results now
        rag_threshold = 0.1 # Apply threshold here for context
        filtered_rag_results = [point for point in text_results if point.score >= rag_threshold]
        print(f"ℹ️ RAG Context: Using {len(filtered_rag_results)} chunks (score >= {rag_threshold}) out of {len(text_results)} retrieved.")
        if not filtered_rag_results:
             if text_results: print(f"ℹ️ No chunks above RAG threshold ({rag_threshold}). Max score: {text_results[0].score:.4f}")
             else: print(f"ℹ️ No chunks retrieved at all.")
             return "No relevant information found in the resume(s)."
        seen_texts = set(); relevant_texts = []
        for point in sorted(filtered_rag_results, key=lambda x: x.score, reverse=True):
            text = point.payload.get("text", ""); doc_id = point.payload.get("document_id", "N/A"); cand_id = point.payload.get("candidate_id", "N/A"); cand_name = point.payload.get("candidate_name", "N/A"); score = point.score; page = point.payload.get("page", "N/A"); chunk_idx = point.payload.get("chunk_index", "N/A")
            context_prefix = f"[Source: {doc_id}, Cand: {cand_name} ({cand_id}), Page: {page}, Chunk: {chunk_idx}, Score: {score:.3f}]\n"
            full_text = context_prefix + text
            if text not in seen_texts: relevant_texts.append(full_text); seen_texts.add(text)
        if not relevant_texts: return "No relevant information found."
        text_context = "\n\n---\n\n".join(relevant_texts)
        return f"Context from Resume(s):\n---------------------\n{text_context}\n---------------------"

    def query(self, query, candidate_id=None, candidate_name=None):
        print(f"\n⏳ Generating context for query: '{query}'" + (f" for cand_id: '{candidate_id}'" if candidate_id else ""))
        context = self.generate_context(query=query, candidate_id=candidate_id)
        if not context or context.strip().startswith("No relevant information found"):
            print("❌ No relevant context found above RAG threshold."); no_context_prompt = f"""User question: "{query}". Candidate: ID='{candidate_id}', Name='{candidate_name}'. No relevant excerpts found in resume documents meeting relevance threshold. Inform user concisely."""
            return self.call_llm(no_context_prompt, is_no_context=True)
        print(f"✅ Context generated. Querying LLM: {self.model_name}")
        candidate_prompt_info = f"Candidate Info: ID={candidate_id}, Name={candidate_name}\n\n" if candidate_id else ""
        qa_prompt_tmpl_str = f"""**Resume Query Analysis**
        **User Question:** "{query}"
        {candidate_prompt_info}**Extracted Resume Information (Context):**\n{context}
        **Instructions:** Answer based *only* on the provided context. Be concise. If info is missing, state that.
        **Answer:**"""
        return self.call_llm(qa_prompt_tmpl_str)

    def call_llm(self, prompt, is_no_context=False):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        system_message = "You are a helpful assistant analyzing resume content based strictly on provided excerpts. If info is missing, state that."
        if is_no_context: system_message = "You are a helpful assistant. Inform user concisely that requested info wasn't found in resume excerpts based on relevance."
        payload = {"model": self.model_name, "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 1500, "top_p": 0.9}
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=120); response.raise_for_status(); result = response.json()
            if result.get("choices") and result["choices"][0].get("message") and result["choices"][0]["message"].get("content"): print("✅ LLM response received."); return result["choices"][0]["message"]["content"].strip()
            elif "error" in result: print(f"❌ Groq API error: {result['error'].get('message')}"); return f"Error: AI API error - {result['error'].get('message')}"
            else: print(f"❌ Unexpected Groq API response: {result}"); return "Error: Unexpected AI API response."
        except requests.exceptions.Timeout: print(f"❌ Groq API timed out."); return "Error: AI API request timed out."
        except requests.exceptions.RequestException as e: print(f"❌ Groq API call error: {e}"); error_detail = e.response.text if e.response else "N/A"; print(f"❌ Error details: {error_detail}"); return f"Error: AI API comms failed. {e}"
        except Exception as e: print(f"❌ Unexpected LLM query error: {e}"); return f"Error: Unexpected error. {e}"


# --- Main Execution (MODIFIED process_resumes call) ---
def process_resumes(directory_path, collection_name="resumes_db", force_recreate_collection=False):
    pdf_paths = glob.glob(os.path.join(directory_path, "*.pdf"))
    if not pdf_paths: print(f"❌ No PDF files found: {directory_path}"); return None, None
    print(f"Found {len(pdf_paths)} PDF files.")

    # Initialize with paragraph chunking strategy
    pdf_processor = PDFProcessor(chunk_strategy='paragraph')
    vector_db = ResumeVectorDB(collection_name=collection_name, text_vector_dim=pdf_processor.embedding_dim)
    vector_db.define_client()
    if not vector_db.client: print("❌ Halting: Qdrant client failed."); return None, None
    vector_db.create_collection(force_recreate=force_recreate_collection) # Force recreate

    processed_count = 0; ingested_count = 0; candidate_map = {}
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path); document_id = os.path.splitext(filename)[0]
        print(f"\n--- Processing: {filename} ---")
        processed_count += 1
        # Extract chunks (list of dicts) and raw first page text
        chunks, raw_first_page = pdf_processor.extract_from_pdf(pdf_path)
        if not chunks: print(f"⚠️ No chunks extracted from {filename}, skipping."); continue
        candidate_name = pdf_processor.extract_candidate_name(raw_first_page)
        if candidate_name == "Unknown Candidate": print(f"⚠️ No name found in {filename}. Using fallback."); candidate_id = document_id.lower().replace(" ", "_")
        else: candidate_id = candidate_name.lower().replace(" ", "_")
        print(f"   - Doc ID: {document_id}, Cand Name: {candidate_name}, Cand ID: {candidate_id}")
        if candidate_id not in candidate_map: candidate_map[candidate_id] = {"name": candidate_name, "docs": []}
        candidate_map[candidate_id]["docs"].append(document_id)
        # Embed based on the 'text' field in each chunk dict
        text_embeddings = pdf_processor.embed_text(chunks)
        if not text_embeddings or len(text_embeddings) != len(chunks): print(f"⚠️ Embedding issue for {filename}, skipping ingest."); continue
        # Pass the list of chunk dicts to ingestion
        success = vector_db.ingest_text_data(text_embeddings, chunks, document_id, candidate_id, candidate_name)
        if success: ingested_count += 1

    print(f"\n--- Processing Summary ---"); print(f"Processed: {processed_count}, Newly Ingested: {ingested_count}"); print(f"Candidate Map: {candidate_map}")
    retriever = ResumeRetriever(vector_db, pdf_processor); rag = ResumeRAG(retriever)
    return rag, candidate_map

# --- if __name__ == "__main__": (Keep as before, ensure force_recreate=True) ---
if __name__ == "__main__":
    start_time = time.time()
    resumes_directory = "/Users/ShahYash/Desktop/Projects/ask-my-prof/AskMyProf/frontend/rag/test_resumes"
    qdrant_collection = "candidate_resumes_v2" # Keep same name, rely on force_recreate
    force_recreate = True # <<<--- Ensure this is True for testing
    print(f"Using Qdrant Collection: {qdrant_collection} (Force Recreate: {force_recreate})")

    if not os.path.isdir(resumes_directory): print(f"❌ Error: Directory not found: {resumes_directory}")
    else:
        print("🚀 Starting Resume Processing and RAG Setup...")
        resume_rag, candidates = process_resumes(resumes_directory, collection_name=qdrant_collection, force_recreate_collection=force_recreate)
        if resume_rag and candidates:
            print("\n✅ RAG System Ready for Queries.")
            target_candidate_id = "yash_shah"; target_candidate_name = candidates.get(target_candidate_id, {}).get("name", "Unknown")
            if target_candidate_name != "Unknown":
                queries = ["What is the candidate's name and contact details?", "What are the candidate's technical skills?", "Where did Yash Shah intern as an AI Software Engineer?", "List the candidate's projects", "What certifications does the candidate have?"]
                for query in queries:
                     print(f"\n❓ Querying (Candidate '{target_candidate_name}'): {query}"); result = resume_rag.query(query=query, candidate_id=target_candidate_id, candidate_name=target_candidate_name)
                     print(f"\n💬 Answer:"); print(result); print("-" * 30)
            else: print(f"\n⚠️ Cannot run specific queries: Candidate ID '{target_candidate_id}' not found.")
        else: print("❌ Error: Resume RAG system could not be initialized.")
    end_time = time.time(); print(f"\n--- Total Execution Time: {end_time - start_time:.2f} seconds ---")