import requests
import argparse

API_NAME = "verify"

def parse_args():
    parser = argparse.ArgumentParser("Stable Diffusion Inference")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape painting", help="Prompt for the model")
    parser.add_argument("--output-path", type=str, default="output.jpg", help="Path to save the generated image")
    return parser.parse_args()

def main():
    args = parse_args()
    url = f"http://localhost:{args.port}/{API_NAME}"
    json_request = {
        "output_path": args.output_path,
        "prompt": args.prompt,
        "ddim_steps": 30,
        "H": 1024,
        "W": 1024,
        "txhash": "0x343434354",
        "image_path": "./output.jpg",
    }
    response = requests.post(url, json=json_request)
    print(response.text)
    import json
    with open('result.json', 'w') as f:
        json.dump(response.json(), f)

if __name__ == "__main__":
    main()