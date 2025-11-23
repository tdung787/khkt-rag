import requests
import time
from concurrent.futures import ThreadPoolExecutor

def test_request(n):
    start = time.time()
    response = requests.get("http://localhost:8110/health")
    elapsed = time.time() - start
    print(f"✅ Request {n}: {elapsed:.2f}s - Status: {response.status_code}")
    return elapsed

# Test 12 concurrent requests
print("🚀 Testing 12 concurrent requests...\n")
start_total = time.time()

with ThreadPoolExecutor(max_workers=12) as executor:
    results = list(executor.map(test_request, range(1, 13)))

total = time.time() - start_total
print(f"\n📊 Total time: {total:.2f}s")
print(f"📊 Average: {sum(results)/len(results):.2f}s per request")