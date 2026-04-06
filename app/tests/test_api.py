import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "tenure": 12,
    "monthly_charges": 79.99,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic"
}

response = requests.post(url, json=data)
print(response.json())
