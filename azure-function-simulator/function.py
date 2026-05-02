import requests
import time

def call_ml_service(x):
    url = "http://ml-service:8000/predict"

    for i in range(10):  # retry up to 10 times
        try:
            response = requests.get(url, params={"x": x}, timeout=2)
            return response.json()
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(2)

    return {"error": "ML service not available"}

def main(x: float):
    result = call_ml_service(x)

    return {
        "message": "Processed by Azure Function (simulated)",
        "input": x,
        "ml_result": result
    }

if __name__ == "__main__":
    print(main(10))