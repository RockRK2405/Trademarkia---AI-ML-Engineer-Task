import json
import logging
import os
import pickle
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from cache import SemanticCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Trademarkia Semantic Search API")

# Models and State
class QueryRequest(BaseModel):
    query: str

class State:
    def __init__(self):
        self.embedding_model = None
        self.vector_index = None
        self.pca_model = None
        self.gmm_model = None
        self.corpus = None
        self.cache = None

state = State()

# Constants
MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_DIR = 'data'
VECTOR_DB_PATH = os.path.join(DATA_DIR, 'faiss_index.bin')
CORPUS_PATH = os.path.join(DATA_DIR, 'corpus.json')
MODEL_PATH = os.path.join(DATA_DIR, 'gmm_model.pkl')
PCA_MODEL_PATH = os.path.join(DATA_DIR, 'pca_model.pkl')

@app.on_event("startup")
def load_resources():
    logger.info("Starting up and loading resources...")
    
    # 1. Load Sentence Transformer
    if not state.embedding_model:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        state.embedding_model = SentenceTransformer(MODEL_NAME)
        
    # 2. Load FAISS Index
    if not state.vector_index:
        if not os.path.exists(VECTOR_DB_PATH):
            raise RuntimeError(f"FAISS index not found at {VECTOR_DB_PATH}. Run data_prep.py.")
        logger.info("Loading FAISS Index...")
        state.vector_index = faiss.read_index(VECTOR_DB_PATH)
        
    # 3. Load Corpus
    if not state.corpus:
        if not os.path.exists(CORPUS_PATH):
            raise RuntimeError(f"Corpus not found at {CORPUS_PATH}. Run data_prep.py.")
        logger.info("Loading corpus...")
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            state.corpus = json.load(f)

    # 4. Load PCA and GMM Models
    if not state.pca_model or not state.gmm_model:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(PCA_MODEL_PATH):
            raise RuntimeError("Clustering models not found. Run clustering.py.")
        logger.info("Loading PCA and GMM models...")
        with open(PCA_MODEL_PATH, 'rb') as f:
            state.pca_model = pickle.load(f)
        with open(MODEL_PATH, 'rb') as f:
            state.gmm_model = pickle.load(f)

    # 5. Initialize Cache. 
    # Similarity threshold set to 0.75 as a heuristic block for related semantics
    if not state.cache:
        logger.info("Initializing semantic cache...")
        state.cache = SemanticCache(similarity_threshold=0.75)
        
    logger.info("All resources loaded successfully.")

@app.post("/query")
def process_query(req: QueryRequest):
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    # 1. Embed query
    query_embedding = state.embedding_model.encode([query_text], convert_to_numpy=True)
    
    # FAISS inner product index requires L2 normalized vectors for Cosine Similarity
    faiss.normalize_L2(query_embedding)
    # The output from model is (1, D), we need flat 1D array for cosine similarity function later
    query_embedding_flat = query_embedding.flatten()
    
    # 2. Determine Dominant Cluster
    # Reduce dimension via PCA before passing to GMM
    reduced_query = state.pca_model.transform(query_embedding)
    cluster_probs = state.gmm_model.predict_proba(reduced_query)[0]
    dominant_cluster = int(np.argmax(cluster_probs))
    
    # 3. Check Semantic Cache
    cache_hit_response = state.cache.search(query_text, query_embedding_flat, dominant_cluster)
    
    if cache_hit_response:
        return cache_hit_response
        
    # 4. Cache Miss - Compute result using FAISS Vector DB
    # We retrieve the top 3 most similar documents
    k = 3
    distances, indices = state.vector_index.search(query_embedding, k)
    
    # Format the result based on retrieved docs
    retrieved_docs = []
    for rank, idx in enumerate(indices[0]):
        # Distance output for FlatIP on normalized vectors is the cosine similarity score
        score = float(distances[0][rank])
        doc = state.corpus[int(idx)]
        retrieved_docs.append({
            "score": round(score, 3),
            "category": doc["original_category_name"],
            "snippet": doc["text"][:200] + "..." # Just returning a snippet for brevity
        })
        
    result_data = {
        "status": "Computed from Vector DB",
        "top_results": retrieved_docs
    }
    
    # Store into cache
    state.cache.insert(query_text, query_embedding_flat, dominant_cluster, result_data)
    
    # 5. Return miss response 
    # (Matches spec from Part 4: on miss compute, store, and return)
    return {
        "query": query_text,
        "cache_hit": False,
        "similarity_score": 1.0, # A miss implicitly has a score of 1.0 against itself
        "result": result_data,
        "dominant_cluster": dominant_cluster,
        "cluster_probabilities": [round(float(p), 4) for p in cluster_probs]
    }

@app.get("/cache/stats")
def get_cache_stats():
    return state.cache.get_stats()

@app.delete("/cache")
def flush_cache():
    state.cache.flush()
    return {"message": "Cache flushed successfully."}

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run ensures it starts cleanly with a single command
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
