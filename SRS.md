# Software Requirements Specification (SRS)
## Trademarkia AI/ML Engineer Task

### 1. Introduction
#### 1.1 Purpose
The purpose of this document is to define the requirements and architectural design for a lightweight semantic search system over the 20 Newsgroups dataset. It includes a custom semantic cache and a FastAPI layer for interacting with the system.

#### 1.2 Scope
The system will process the dataset, generate vector embeddings, run fuzzy clustering to uncover latent structures, and expose a semantic search endpoint that utilizes a custom-built, cluster-aware semantic cache without external dependencies like Redis.

### 2. Overall Description
#### 2.1 System Features
*   **Data Ingestion & Embedding**: Clean and process the 20 Newsgroups corpus, embedding the text into a dense vector space using a local Transformer model.
*   **Vector Storage**: Store embeddings in a localized Vector DB (FAISS) for quick retrieval.
*   **Fuzzy Clustering**: Perform soft clustering (Gaussian Mixture Model) mapping texts to cluster probability distributions rather than hard labels.
*   **Semantic Caching**: A memory-resident cache that evaluates query semantics to avoid redundant computations on similarly phrased queries.
*   **RESTful API**: FastAPI server exposing search, cache statistics, and cache invalidation.

#### 2.2 Operating Environment
*   **Software**: Python 3.9+, Docker.
*   **Dependencies**: FastAPI, Uvicorn, scikit-learn, sentence-transformers, faiss-cpu, numpy.

### 3. System Architecture & Components
#### 3.1 Embedding & Vector DB (Part 1)
*   **Data Prep**: Remove metadata (headers, footers, quotes) to focus on raw semantic content.
*   **Model**: `all-MiniLM-L6-v2` for generating 384-dimensional embeddings. It offers a strong balance of speed and semantic quality.
*   **Storage**: FAISS Index for $L2$ or Cosine Similarity search.

#### 3.2 Fuzzy Clustering (Part 2)
*   **Method**: Gaussian Mixture Model (GMM).
*   **Validation**: The number of clusters $K$ will be selected based on analyzing the Bayesian Information Criterion (BIC) or Silhouette Scores.
*   **Output**: For any document $i$, the model yields $P(C_k | i)$, representing its probability distribution across $K$ clusters.

#### 3.3 Semantic Cache (Part 3)
*   **Structure**: A dictionary of lists, keyed by cluster ID.
*   **Algorithm**: When a query $Q$ arrives, find its dominant cluster $C_d$. Search for previous queries within the bucket for $C_d$. If $\text{cosine\_sim}(Q, Q_{cached}) > \theta$, return cached result.
*   **Tunable Parameter**: $\theta$ (similarity threshold). A larger $\theta$ increases precision (fewer false hits) but decreases recall (lower cache hit rate). Evaluating this trade-off is critical.

#### 3.4 API Layer (Part 4)
*   `POST /query`: Primary endpoint for search.
*   `GET /cache/stats`: Diagnostic endpoint.
*   `DELETE /cache`: Maintenance endpoint.
