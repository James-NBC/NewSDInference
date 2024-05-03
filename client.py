# File name: model_client.py
import requests

json_request = {
    "prompt": "A beautiful landscape painting",
    "seed": 10,
    "output_path": "output.jpg",
}

response = requests.post("http://127.0.0.1:8000/", json=json_request)
json_respone = response.json

print(json_respone)