import logging
import numpy as np

logger = logging.getLogger(__name__)

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two 1D normalized arrays."""
    return np.dot(v1, v2)

class SemanticCache:
    """
    A custom semantic cache structure.
    
    Instead of checking a new query against *all* cached queries (O(N) lookup), 
    we check the dominant cluster of the new query, and only compare against 
    cached queries within that same cluster bucket.
    
    This reduces lookup time complexity to approximately O(N/K), where K is the 
    number of clusters, assuming queries are somewhat evenly distributed.
    """
    
    def __init__(self, similarity_threshold=0.75):
        """
        Args:
            similarity_threshold (float): The tunable threshold.
                - Higher (e.g., 0.95): Lower hit rate, but hits are guaranteed to be extremely relevant.
                - Lower (e.g., 0.70): Higher hit rate, but risks returning irrelevant answers for loosely related queries.
                0.75 serves as a good default heuristic for SentenceTransformers.
        """
        self.similarity_threshold = similarity_threshold
        # Cache storage: dictionary where keys are cluster IDs, values are lists of cached items
        self.cache = {}
        
        # Stats tracking
        self.hits = 0
        self.misses = 0
        self.total_entries = 0
        
        logger.info(f"Initialized Semantic Cache with threshold: {self.similarity_threshold}")

    def _get_bucket(self, cluster_id):
        if cluster_id not in self.cache:
            self.cache[cluster_id] = []
        return self.cache[cluster_id]

    def search(self, query_text: str, query_embedding: np.ndarray, dominant_cluster: int):
        """
        Search for a semantically similar query in the cache.
        """
        bucket = self._get_bucket(dominant_cluster)
        
        best_match = None
        highest_score = -1.0
        
        # Linear search within the specific cluster's bucket
        for item in bucket:
            score = cosine_similarity(query_embedding, item['embedding'])
            if score > highest_score:
                highest_score = score
                best_match = item
                
        if highest_score >= self.similarity_threshold:
            self.hits += 1
            # Return result without the embedding array for cleanliness
            return {
                "cache_hit": True,
                "matched_query": best_match['query'],
                "similarity_score": float(highest_score),
                "result": best_match['result'],
                "dominant_cluster": dominant_cluster
            }
            
        self.misses += 1
        return None

    def insert(self, query_text: str, query_embedding: np.ndarray, dominant_cluster: int, result: str):
        """
        Store a new query and its result into the appropriate cluster bucket.
        """
        bucket = self._get_bucket(dominant_cluster)
        bucket.append({
            "query": query_text,
            "embedding": query_embedding,
            "result": result
        })
        self.total_entries += 1

    def get_stats(self):
        """Returns the current state of the cache."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hits,
            "miss_count": self.misses,
            "hit_rate": round(hit_rate, 3)
        }

    def flush(self):
        """Flushes the cache entirely and resets statistics."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.total_entries = 0
        logger.info("Semantic cache flushed.")
