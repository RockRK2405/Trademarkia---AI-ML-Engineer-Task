import json
import logging
import os
import faiss
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

import ssl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bypass SSL Verification locally to download datasets smoothly (macOS fix)
ssl._create_default_https_context = ssl._create_unverified_context


# Constants
MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_DIR = 'data'
VECTOR_DB_PATH = os.path.join(DATA_DIR, 'faiss_index.bin')
CORPUS_PATH = os.path.join(DATA_DIR, 'corpus.json')
# We remove headers, footers, and quotes to prevent the model from clustering
# based on metadata (like email addresses or organizations) instead of semantic content.
REMOVE_METADATA = ('headers', 'footers', 'quotes')

def prepare_data():
    """
    Downloads, cleans, and prepares the 20 Newsgroups dataset.
    Generates embeddings and stores them in a FAISS index.
    Saves the cleaned corpus as a JSON file for retrieval.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("Fetching 20 Newsgroups dataset...")
    # We fetch both train and test subsets to have a comprehensive corpus.
    # We remove metadata to focus purely on the text semantics.
    newsgroups = fetch_20newsgroups(subset='all', remove=REMOVE_METADATA)
    
    documents = newsgroups.data
    targets = newsgroups.target
    target_names = newsgroups.target_names

    logger.info(f"Loaded {len(documents)} documents.")

    # Filter out empty or extremely short documents (less than 20 characters after stripping)
    # as they carry little semantic value and add noise to clustering.
    logger.info("Cleaning corpus...")
    cleaned_corpus = []
    skipped = 0
    for i, doc in enumerate(documents):
        cleaned_text = doc.strip()
        if len(cleaned_text) < 20:
            skipped += 1
            continue
        cleaned_corpus.append({
            "id": len(cleaned_corpus),
            "text": cleaned_text,
            "original_category_id": int(targets[i]),
            "original_category_name": target_names[targets[i]]
        })
    
    logger.info(f"Retained {len(cleaned_corpus)} documents. Skipped {skipped} noisy/empty documents.")

    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    logger.info("Generating embeddings (this may take a few minutes)...")
    texts_to_embed = [doc["text"] for doc in cleaned_corpus]
    # We use a batch size of 256. all-MiniLM-L6-v2 is fast and lightweight enough to handle this on CPU/GPU.
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, batch_size=256, convert_to_numpy=True)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")

    logger.info("Normalizing embeddings for cosine similarity...")
    # FAISS inner product index behaves exactly like cosine similarity if vectors are L2 normalized
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    logger.info(f"Building FAISS Index (IndexFlatIP) for {dimension} dimensions...")
    # IndexFlatIP calculates exact inner product. Given the dataset size (~18k), 
    # exact search is extremely fast in FAISS and we don't need approximate methods like HNSW or IVF which add complexity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    logger.info(f"Saving FAISS index to {VECTOR_DB_PATH}")
    faiss.write_index(index, VECTOR_DB_PATH)

    logger.info(f"Saving corpus mapping to {CORPUS_PATH}")
    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(cleaned_corpus, f, indent=2)

    logger.info("Data preparation complete!")

if __name__ == "__main__":
    prepare_data()
