import os
import cv2
import time
import torch
import tempfile 
import argparse
import numpy as np
from PIL import Image 
import onnxruntime as ort
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
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipeline = load_pipeline(CKPT_DIR, device)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(os.path.join(CKPT_DIR, "safety_checker")).to("cuda")
feature_extractor = CLIPFeatureExtractor.from_pretrained(os.path.join(CKPT_DIR, "feature_extractor"))
onnx_model = ort.InferenceSession(os.path.join(CKPT_DIR, "verifier.onnx"))

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
    requested_height: int = 512,
    requested_width: int = 512,
    requested_ddim_steps: int = 30,
    requested_seed: int = 42,
):
    seed_everything(requested_seed)
    pil_images = pipeline(
        prompt,
        height=requested_height,
        width=requested_width,
        guidance_scale=2.0,
        num_images_per_prompt=1,
        num_inference_steps=requested_ddim_steps,
    ).images  
    # pil to torch
    has_nsfw_concepts = check_nsfw_images(pil_images)
    checked_image = pil_images[0]
    # reshape from 1024x1024 to 512x512
    if requested_height != 512:
        checked_image = checked_image.resize((512, 512))
    if has_nsfw_concepts[0]:
        checked_image = Image.new("RGB", (512, 512), (0, 0, 0))
    torch.cuda.empty_cache()
    return checked_image

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
    generated_image = to_generate(prompt, requested_height, requested_width, requested_ddim_steps, requested_seed)
    generated_image.save(output_path)
    return jsonify({"output_path": output_path, "time": time.time() - start, "seed": requested_seed})  

@app.route('/verify', methods=['POST'])
def verify():
    json_request = request.get_json(force=True)
    prompt = json_request['prompt']
    requested_height = json_request['H']
    requested_width = json_request['W']
    requested_ddim_steps = json_request['ddim_steps']
    requested_tx_hash = json_request['txhash'][2:]
    to_verify_image_path = json_request['image_path']
    requested_seed = int(requested_tx_hash, 16) % (2**32)
    # euclid distance
    generated_image = to_generate(prompt, requested_height, requested_width, requested_ddim_steps, requested_seed)
    temp_file_path = tempfile.mkstemp(suffix= os.path.basename(to_verify_image_path))[1]
    generated_image.save(temp_file_path)
    generated_image = cv2.imread(temp_file_path)
    verify_image = cv2.imread(to_verify_image_path)
    generated_image = (np.transpose(generated_image, (2, 0, 1))).astype(np.float32)
    verify_image = (np.transpose(verify_image, (2, 0, 1))).astype(np.float32)
    generated_image = np.expand_dims(generated_image, axis=0)
    verify_image = np.expand_dims(verify_image, axis=0)
    output1 = onnx_model.run(None, {'input': generated_image})[0]
    output2 = onnx_model.run(None, {'input': verify_image})[0]
    # similarity score
    similarity = np.dot(output1, output2.T) / (np.linalg.norm(output1) * np.linalg.norm(output2))
    similarity = max(float(similarity), 1.0)
    if similarity > 0.995:
        return jsonify({"verified": True, "similarity": float(similarity)})
    return jsonify({"verified": False, "similarity": float(similarity)})


if __name__ == "__main__":
    args = parse_args()
    app.run(host='127.0.0.1', port = args.port)