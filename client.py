# File name: model_client.py
import requests

json_request = {
    "input": {
        "prompt": "A strong man",
        "seed": 10,
        "steps": 2,
    }
}

response = requests.post("https://sdxl-max-multi.eternalai.org:5000/predictions", json=json_request)
json_respone = response.json()

with open("dummy_result.json", "wb") as f:
    f.write(response.content)