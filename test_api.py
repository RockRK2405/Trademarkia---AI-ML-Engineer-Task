import requests
import json
import time

API_URL = "http://127.0.0.1:8000"

def test_api():
    print("Checking if API is up...")
    try:
        requests.get(f"{API_URL}/docs")
    except requests.exceptions.ConnectionError:
        print("API is not running! Start it with `uvicorn main:app --reload`")
        return

    print("\n--- 1. Flushing Cache ---")
    res = requests.delete(f"{API_URL}/cache")
    print(res.json())

    query_1 = "Why are NASA budgets always being cut?"
    
    print(f"\n--- 2. Query 1 (Expected Miss) ---")
    print(f"Query: '{query_1}'")
    start = time.time()
    res = requests.post(f"{API_URL}/query", json={"query": query_1})
    duration = time.time() - start
    data = res.json()
    print(f"Time: {duration:.4f}s | Cache Hit: {data.get('cache_hit')} | Score: {data.get('similarity_score')}")
    print(f"Cluster: {data.get('dominant_cluster')}")
    
    # query_2 is semantically very similar to query_1
    query_2 = "What is the reason behind the constant reduction in funding for NASA?"
    
    print(f"\n--- 3. Query 2 (Expected Hit) ---")
    print(f"Query: '{query_2}'")
    start = time.time()
    res = requests.post(f"{API_URL}/query", json={"query": query_2})
    duration = time.time() - start
    data = res.json()
    print(f"Time: {duration:.4f}s | Cache Hit: {data.get('cache_hit')} | Score: {data.get('similarity_score')}")
    print(f"Matched Cached Query: '{data.get('matched_query', 'N/A')}'")
    
    query_3 = "Should civilians be allowed to own assault rifles?"
    
    print(f"\n--- 4. Query 3 (Expected Miss - completely different topic) ---")
    print(f"Query: '{query_3}'")
    start = time.time()
    res = requests.post(f"{API_URL}/query", json={"query": query_3})
    duration = time.time() - start
    data = res.json()
    print(f"Time: {duration:.4f}s | Cache Hit: {data.get('cache_hit')} | Score: {data.get('similarity_score')}")
    print(f"Cluster: {data.get('dominant_cluster')}")

    print("\n--- 5. Checking Stats ---")
    res = requests.get(f"{API_URL}/cache/stats")
    print(json.dumps(res.json(), indent=2))

if __name__ == "__main__":
    test_api()
