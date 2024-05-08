# File name: model_client.py
import requests

json_request = {
    "prompt": "A beautiful landscape painting",
    "seed": 10,
    "output_path": "output.jpg",
    "steps": 5,
}

response = requests.post("http://127.0.0.1:8000/", json=json_request)
json_respone = response.json()

with open("dummy_result.json", "wb") as f:
    f.write(response.content)