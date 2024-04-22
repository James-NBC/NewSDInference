import os
import cv2
import time
import torch
import argparse
import numpy as np
import onnxruntime
from PIL import Image 
from flask import Flask, request, jsonify
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline
from transformers import CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker


app = Flask(__name__)

def load_pipeline(ckpt_dir, device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DiffusionPipeline.from_pretrained(os.path.join(ckpt_dir, "sdxl_lightning")).to(device)
    return pipeline

def parse_args():
    parser = argparse.ArgumentParser("Stable Diffusion Inference")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    return parser.parse_args()

CKPT_DIR = "checkpoints"
VERIFIER_PATH = os.path.join(CKPT_DIR, "verifier.onnx")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipeline = load_pipeline(CKPT_DIR, device)
verifier = onnxruntime.InferenceSession(VERIFIER_PATH)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(os.path.join(CKPT_DIR, "safety_checker")).to("cuda")
feature_extractor = CLIPFeatureExtractor.from_pretrained(os.path.join(CKPT_DIR, "feature_extractor"))

EPS = 1e-6

@app.route('/')
def index():
    return "Hello, World!"

def check_nsfw_images(images: list[Image.Image]):
    safety_checker_input = feature_extractor(images, return_tensors="pt").to("cuda")
    images_np = [np.array(img) for img in images]

    _, has_nsfw_concepts = safety_checker(
        images=images_np,
        clip_input=safety_checker_input.pixel_values.to("cuda"),
    )
    
    return has_nsfw_concepts

def to_generate(
    prompt: str = "A painting of a beautiful sunset over a calm lake",
    output_path: str = "output.png",
    requested_height: int = 512,
    requested_width: int = 512,
    requested_ddim_steps: int = 30,
    requested_seed: int = 42,
):
    seed_everything(requested_seed)
    pil_images = pipeline(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=2.0,
        num_images_per_prompt=1,
        num_inference_steps=requested_ddim_steps,
    ).images  
    # pil to torch
    has_nsfw_concepts = check_nsfw_images(pil_images)
    checked_image = pil_images[0]
    # reshape from 1024x1024 to 512x512
    checked_image = checked_image.resize((requested_width, requested_height))
    if has_nsfw_concepts[0]:
        checked_image = Image.new("RGB", (requested_width, requested_height), (0, 0, 0))
    checked_numpy_image = np.array(checked_image).transpose(2, 0, 1)/255.0
    # pil to numpy
    checked_numpy_image = np.expand_dims(checked_numpy_image, axis=0)
    verified_embedding = verifier.run(None, {'input': checked_numpy_image.astype(np.float32)})[0].tolist()
    checked_image.save(output_path)
    torch.cuda.empty_cache()
    return verified_embedding

@app.route('/generate_image', methods=['POST'])
def generate_image():
    json_request = request.get_json(force=True)
    prompt = json_request['prompt']
    output_path = json_request['output_path']
    requested_height = json_request['H']
    requested_width = json_request['W']
    requested_ddim_steps = json_request['ddim_steps']
    requested_tx_hash = json_request['txhash'][2:]
    requested_seed = int(requested_tx_hash, 16) % (2**32)
    start = time.time()
    verified_embedding = to_generate(prompt, output_path, requested_height, requested_width, requested_ddim_steps, requested_seed)
    torch.cuda.empty_cache()
    return jsonify({"output_path": output_path, "embedding": verified_embedding, "time": time.time() - start, "seed": requested_seed})  

@app.route('/verify', methods=['POST'])
def verify():
    json_request = request.get_json(force=True)
    prompt = json_request['prompt']
    output_path = json_request['output_path']
    requested_height = json_request['H']
    requested_width = json_request['W']
    requested_ddim_steps = json_request['ddim_steps']
    requested_tx_hash = json_request['txhash'][2:]
    embedding = json_request['embedding']
    requested_seed = int(requested_tx_hash, 16) % (2**32)
    # euclid distance
    verified_embedding = to_generate(prompt, output_path, requested_height, requested_width, requested_ddim_steps, requested_seed)
    distance = np.linalg.norm(np.array(embedding) - np.array(verified_embedding))
    if distance < EPS:
        return jsonify({"verified": True, "distance": distance, "verified_embedidng": verified_embedding})
    return jsonify({"verified": False, "distance": distance, "verified_embedidng": verified_embedding})

if __name__ == "__main__":
    args = parse_args()
    app.run(host='127.0.0.1', port = args.port)