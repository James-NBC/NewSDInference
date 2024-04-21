import requests
import argparse
import random

API_NAME = "generate_image"

def parse_args():
    parser = argparse.ArgumentParser("Stable Diffusion Inference")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape painting", help="Prompt for the model")
    parser.add_argument("--output-path", type=str, default="output.jpg", help="Path to save the generated image")
    return parser.parse_args()

def main():
    args = parse_args()
    url = f"http://localhost:{args.port}/{API_NAME}"
    random_seed = random.randint(0, 2**32 - 1)
    json_request = {
        "output_path": args.output_path,
        "prompt": args.prompt,
        "ddim_steps": 30,
        "H": 512,
        "W": 512,
        "txhash": "0x343434354",
    }
    response = requests.post(url, json=json_request)
    import json
    with open('result.json', 'w') as f:
        json.dump(response.json(), f)

if __name__ == "__main__":
    main()