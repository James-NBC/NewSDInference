import os
import cv2
import time
import torch
import argparse
import numpy as np
import onnxruntime
from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline
from flask import Flask, request, jsonify


app = Flask(__name__)

def load_pipeline(ckpt_dir, device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DiffusionPipeline.from_pretrained(ckpt_dir).to(device)
    return pipeline

def parse_args():
    parser = argparse.ArgumentParser("Stable Diffusion Inference")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    return parser.parse_args()

CKPT_DIR = "data/checkpoints"
SD_CKPT_DIR = os.path.join(CKPT_DIR, "sd_ckpt")
VERIFIER_PATH = os.path.join(CKPT_DIR, "verifier.onnx")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipeline = load_pipeline(SD_CKPT_DIR, device)
verifier = onnxruntime.InferenceSession(VERIFIER_PATH)


@app.route('/')
def index():
    return "Hello, World!"

@app.route('/generate_image', methods=['POST'])
def generate_image():
    seed_everything(42)
    json_request = request.get_json(force=True)
    prompt = json_request['prompt']
    output_path = json_request['output_path']
    requested_H = json_request['H']
    requested_W = json_request['W']
    requested_ddim_steps = json_request['ddim_steps']
    requested_scale = json_request['scale']
    start = time.time()
    checked_image_torch = pipeline(
        prompt,
        height=requested_H,
        width=requested_W,
        num_images_per_prompt=1,
        num_inference_steps=requested_ddim_steps,
        guidance_scale=requested_scale,
        output_type = "pt",
    ).images   
    # change dim order from B, H, W, C to B, C, H, W
    checked_image_numpy = checked_image_torch.cpu().numpy()
    verified_embedding = verifier.run(None, {'input': checked_image_numpy.astype(np.float32)})[0].tolist()
    checked_image = checked_image_numpy[0].transpose(1, 2, 0)
    open_cv_image = (checked_image * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR))
    return jsonify({"output_path": output_path, "embedding": verified_embedding, "time": time.time() - start})  

if __name__ == "__main__":
    args = parse_args()
    app.run(host='127.0.0.1', port = args.port)