# Trademarkia AI/ML Engineer Task - Semantic Cache System

This repository contains my solution for the Trademarkia AI/ML Engineer assignment. It implements a fully containerized, lightweight Semantic Search system built over the [20 Newsgroups Dataset](https://archive.ics.uci.edu/dataset/113/twenty+newsgroups).

## Project Overview

The core objective of this project is to create an efficient, memory-resident semantic cache capable of detecting when a rephrased query holds the exact same semantic intent as a previously computed one. 

This system contains three main pillars:
1. **Embedding & Vector Storage**: Raw datasets are cleaned (stripping meta-noise like headers and quotes), embedded using `sentence-transformers/all-MiniLM-L6-v2`, and stored via `FAISS` using an exact L2 inner product (Cosine Similarity) index.
2. **Fuzzy Clustering**: Using PCA dimensionality reduction and a Gaussian Mixture Model (GMM), the corpus is fuzzily grouped into 20 overlapping clusters. Every document (and incoming query) holds a soft probability distribution depicting its relation across all topics. 
3. **Custom Semantic Cache**: Instead of computing $O(N)$ linear scans against cached items, this cache achieves ~$O(N/K)$ lookup efficiency. Queries are partitioned into memory "buckets" defined by their dominant GMM cluster. Matching thresholds are carefully tuned (Cosine > 0.75) to ensure precision while boosting cache hit rates on similar queries.

## Getting Started

### Local Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the data preparation script
python data_prep.py

# Run the clustering script
python clustering.py

# Start the API
uvicorn main:app --reload
```
### Docker
```bash
docker-compose up --build
```

You can view the real-time API documentation and test the `/query`, `/cache/stats`, and `/cache` deletion endpoints at `http://localhost:8000/docs`. A full breakdown of the architecture choices is available in `SRS.md`.
