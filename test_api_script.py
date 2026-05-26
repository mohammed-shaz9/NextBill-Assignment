import urllib.request
import json

def run_test_endpoint(url, data=None):
    try:
        req = urllib.request.Request(url, method='GET' if data is None else 'POST')
        req.add_header('x-api-key', 'nextbill_dev_secret_key_2026')
        if data is not None:
            req.add_header('Content-Type', 'application/json')
            data = json.dumps(data).encode('utf-8')
        
        with urllib.request.urlopen(req, data=data) as response:
            result = json.loads(response.read().decode('utf-8'))
            print(f"\n[OK] SUCCESS: {url}")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n[ERROR] ERROR: {url} -> {e}")

print("Testing API Endpoints...")

# Test 1: Health check
run_test_endpoint('http://127.0.0.1:8000/health')

# Test 2: Single prediction (Logistics)
run_test_endpoint('http://127.0.0.1:8000/predict', {"text": "Blue Dart courier charges for warehouse delivery"})

# Test 3: Single prediction (Cloud/Software)
run_test_endpoint('http://127.0.0.1:8000/predict', {"text": "AWS monthly cloud hosting bill"})

# Test 4: Batch prediction
run_test_endpoint('http://127.0.0.1:8000/predict/batch', {
    "texts": [
        "Flight tickets to Mumbai for client meeting",
        "Purchase of raw materials from supplier",
        "Monthly electricity bill for the office"
      ]
})
