# test_rag.py
import requests
import time
from concurrent.futures import ThreadPoolExecutor

def test_rag(n):
    start = time.time()
    response = requests.post(
        "http://localhost:8110/api/rag/query",
        data={
            "user_input": "Giải thích định luật Newton thứ 2",
            "session_id": f"sess_20251123145246_f2dd91e5",
            "student_id": "691142b6c9543ec5021f546b"
        }
    )
    elapsed = time.time() - start
    print(f"✅ RAG Request {n}: {elapsed:.2f}s - Status: {response.status_code}")
    return elapsed

print("🚀 Testing 5 concurrent RAG queries...\n")
start = time.time()

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(test_rag, range(1, 6)))

print(f"\n📊 Total: {time.time() - start:.2f}s")
print(f"📊 Avg: {sum(results)/len(results):.2f}s")