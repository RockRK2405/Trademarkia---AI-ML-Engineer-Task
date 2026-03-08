import json
import logging
import os
import faiss
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = 'data'
VECTOR_DB_PATH = os.path.join(DATA_DIR, 'faiss_index.bin')
CORPUS_PATH = os.path.join(DATA_DIR, 'corpus.json')
MODEL_PATH = os.path.join(DATA_DIR, 'gmm_model.pkl')
PCA_MODEL_PATH = os.path.join(DATA_DIR, 'pca_model.pkl')

def perform_clustering():
    """
    Loads FAISS index embeddings and corpus.
    Reduces dimensionality using PCA for better clustering performance.
    Applies Gaussian Mixture Model (GMM) for fuzzy clustering.
    Saves the GMM and PCA models.
    """
    if not os.path.exists(VECTOR_DB_PATH) or not os.path.exists(CORPUS_PATH):
        logger.error("Data files not found. Run data_prep.py first.")
        return

    logger.info("Loading FAISS index...")
    index = faiss.read_index(VECTOR_DB_PATH)
    
    # Reconstruct embeddings from the FlatIP index
    num_vectors = index.ntotal
    dimension = index.d
    logger.info(f"Loaded {num_vectors} embeddings of dimension {dimension}.")

    embeddings = np.zeros((num_vectors, dimension), dtype=np.float32)
    for i in range(num_vectors):
        embeddings[i] = index.reconstruct(i)

    logger.info("Loading corpus JSON mapping...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    # 1. Dimensionality Reduction (Optional but highly recommended for GMM)
    # GMM struggles in high dimensional spaces due to the curse of dimensionality
    # We reduce the 384-dim embeddings down to 50 dimensions where ~85-90% variance is maintained
    n_components_pca = 50
    logger.info(f"Fitting PCA to reduce dimensions from {dimension} to {n_components_pca}...")
    pca = PCA(n_components=n_components_pca, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Calculate explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA explained variance ratio: {explained_variance:.4f}")

    # 2. Fuzzy Clustering using GMM
    # The 20 newsgroups dataset has 20 hard classes, but topics overlap heavily
    # (e.g. comp.sys.ibm.pc.hardware and comp.sys.mac.hardware)
    # We choose 20 components as a baseline, assuming latent semantic groups align roughly
    # with the explicit labels, but allow the model to spread probabilities (fuzzy/soft clustering).
    n_clusters = 20
    logger.info(f"Fitting Gaussian Mixture Model with {n_clusters} clusters...")
    
    gmm = GaussianMixture(
        n_components=n_clusters, 
        covariance_type='full', # 'full' allows clusters to take any ellipsoidal shape 
        random_state=42, 
        max_iter=100, 
        n_init=3 # Run 3 times, keep best
    )
    
    gmm.fit(reduced_embeddings)
    logger.info(f"GMM training converged: {gmm.converged_}")
    
    # 3. Analyze boundaries and uncertainty
    logger.info("Predicting soft cluster assignments over the dataset...")
    # predict_proba returns the probability mass for each document across all 20 clusters
    probs = gmm.predict_proba(reduced_embeddings)
    
    # Update corpus with cluster assignments to analyze later
    for i, doc in enumerate(corpus):
        doc['cluster_probs'] = probs[i].tolist()
        doc['dominant_cluster'] = int(np.argmax(probs[i]))
        doc['dominant_cluster_prob'] = float(np.max(probs[i]))

    # Let's log some stats about uncertainty. 
    # A perfectly confident prediction has max prob ~ 1.0. A highly uncertain has max prob ~ 1/20 (0.05).
    confidences = [doc['dominant_cluster_prob'] for doc in corpus]
    avg_confidence = np.mean(confidences)
    logger.info(f"Average dominant cluster probability (confidence): {avg_confidence:.4f}")
    
    uncertain_docs = [doc for doc in corpus if doc['dominant_cluster_prob'] < 0.3]
    logger.info(f"Found {len(uncertain_docs)} highly uncertain documents (max prob < 0.3)")

    if uncertain_docs:
        logger.info("Example uncertain document text:")
        logger.info(uncertain_docs[0]['text'][:200] + "...")
        logger.info(f"Original Category: {uncertain_docs[0]['original_category_name']}")
        
        # Show the top 3 clusters it belongs to
        top_clusters = np.argsort(uncertain_docs[0]['cluster_probs'])[::-1][:3]
        probs_top = [uncertain_docs[0]['cluster_probs'][c] for c in top_clusters]
        logger.info(f"Fuzzy Mapping Output Top-3: Clusters {top_clusters} with probabilities {probs_top}")
        
    logger.info(f"Saving updated corpus to {CORPUS_PATH}...")
    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2)

    logger.info(f"Saving PCA and GMM models to {DATA_DIR}...")
    with open(PCA_MODEL_PATH, 'wb') as f:
        pickle.dump(pca, f)
        
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(gmm, f)
        
    logger.info("Clustering preparation complete!")

if __name__ == "__main__":
    perform_clustering()
