import requests
import argparse

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
    json_request = {
        "output_path": args.output_path,
        "prompt": args.prompt,
        "ddim_steps": 30,
        "H": 512,
        "W": 512,
        "txhash": "0x71d646c949928e46145dc3bc16af07c5b0861a90d4f707b100c5d97f9158424e55455",
    }
    response = requests.post(url, json=json_request)
    # read image to bytes
    from PIL import Image
    import io
    def image_to_bytes(image_path):
        with open(image_path, 'rb') as img_file:
            img_bytes = img_file.read()
        return img_bytes
    bytes_image = image_to_bytes(args.output_path)
    json_result = response.json()
    json_result["bytes"] = bytes_image
    import json
    with open('result.json', 'w') as f:
        json.dump(json_result, f)

if __name__ == "__main__":
    main()