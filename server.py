import os
import json
import time
import torch
import tempfile 
import argparse
import diffusers
import numpy as np
from PIL import Image 
from open_clip.model import build_model_from_openai_state_dict
from open_clip.transform import PreprocessCfg, image_transform_v2
from flask import Flask, request, jsonify
from transformers import CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

with open('config.json', 'r') as f:
    CONFIG = json.load(f)

app = Flask(__name__)

def load_pipeline(ckpt_dir_or_path, ckpt_type, device = "cuda"):
    if ckpt_type == "safetensors":    
        pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(ckpt_dir_or_path).to(device)
    elif ckpt_type == "lora":
        base_model = CONFIG["base_model"]
        if base_model == "SDXL1.0":
            pipeline = diffusers.StableDiffusionXLPipeline.from_single_file("/home/xxx/base_model/sdxl_1.safetensors").to("cuda")
            pipeline.load_lora_weights(ckpt_dir_or_path)
        else:
            raise ValueError(f"Unknown base model: {base_model}")
    elif ckpt_type == "diffusers":
        pipeline = diffusers.StableDiffusionPipeline.from_pretrained(ckpt_dir_or_path).to(device)
    else:
        raise ValueError(f"Unknown checkpoint type: {ckpt_type}")
    return pipeline

def parse_args():
    parser = argparse.ArgumentParser("Stable Diffusion Inference")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    return parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pipeline = load_pipeline(CONFIG["stable_diffusion_ckpt"], CONFIG["checkpoint_type"], device)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(CONFIG["safety_checker_ckpt"]).to("cuda")
generator = torch.Generator(device=device)
feature_extractor = CLIPFeatureExtractor.from_pretrained(CONFIG["feature_extractor_ckpt"])
preprocess_cfg = {'size': 224, 'mode': 'RGB', 'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711), 'interpolation': 'bicubic', 'resize_mode': 'shortest', 'fill_color': 0}
preprocess = image_transform_v2(
    PreprocessCfg(**preprocess_cfg),
    is_train = False
)
verifier = torch.jit.load(CONFIG["verifier_ckpt"], map_location="cpu").eval()
verifier = build_model_from_openai_state_dict(verifier.state_dict(), cast_dtype = torch.float16).to(device)

@app.route('/')
def index():
    return "Hello, World!"

def check_nsfw_images(images: list[Image.Image]):
    safety_checker_input = feature_extractor(images, return_tensors="pt").to(device)
    images_np = [np.array(img) for img in images]

    _, has_nsfw_concepts = safety_checker(
        images=images_np,
        clip_input=safety_checker_input.pixel_values.to(device),
    )
    
    return has_nsfw_concepts

def to_generate(
    prompt: str = "A painting of a beautiful sunset over a calm lake",
    requested_height: int = 512,
    requested_width: int = 512,
    requested_ddim_steps: int = 30,
    requested_cfg: float = 2.0,
    requested_seed: int = 42,
):
    generator.manual_seed(requested_seed)
    image_latents = torch.randn(
        (1, pipeline.unet.config.in_channels, requested_height // 8, requested_width // 8),
        generator = generator,
        device = "cuda"
    )
    pil_images = pipeline(
        prompt,
        latents = image_latents,
        height=requested_height,
        width=requested_width,
        guidance_scale=requested_cfg,
        num_images_per_prompt=1,
        num_inference_steps=requested_ddim_steps,
    ).images  
    # pil to torch
    has_nsfw_concepts = check_nsfw_images(pil_images)
    checked_image = pil_images[0]
    if has_nsfw_concepts[0]:
        checked_image = Image.new("RGB", (requested_height, requested_width), (0, 0, 0))
    torch.cuda.empty_cache()
    return checked_image

@app.route('/generate_image', methods=['POST'])
def generate_image():
    json_request = request.get_json(force=True)
    prompt = json_request['prompt']
    output_path = json_request['output_path']
    requested_tx_hash = json_request['seed'][2:]
    requested_seed = int(requested_tx_hash, 16) % (2**32)
    start = time.time()
    generated_image = to_generate(prompt, CONFIG["height"], CONFIG["width"], CONFIG["diffusion_steps"], CONFIG["cfg"], requested_seed)
    generated_image.save(output_path, optimize = True, quality = 40)
    return jsonify({"output_path": output_path, "time": time.time() - start, "seed": requested_seed})  

@app.route('/verify', methods=['POST'])
def verify():
    json_request = request.get_json(force=True)
    prompt = json_request['prompt']
    requested_tx_hash = json_request['seed'][2:]
    to_verify_image_path = json_request['image_path']
    requested_seed = int(requested_tx_hash, 16) % (2**32)
    # euclid distance
    generated_image = to_generate(prompt, CONFIG["height"], CONFIG["width"], CONFIG["diffusion_steps"], CONFIG["cfg"], requested_seed)
    temp_file_path = tempfile.mkstemp(suffix= os.path.basename(to_verify_image_path))[1]
    generated_image.save(temp_file_path)
    generated_image = preprocess(Image.open(temp_file_path)).unsqueeze(0).to(device)
    to_verify_image = preprocess(Image.open(to_verify_image_path)).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features_1 = verifier.encode_image(generated_image)
        image_features_2 = verifier.encode_image(to_verify_image)
        image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
        image_features_2 /= image_features_2.norm(dim=-1, keepdim=True)
        similarity = (image_features_1 @ image_features_2.T).mean()
        similarity = max(float(similarity.item()), 1.0)
    torch.cuda.empty_cache()
    if similarity > 0.995:
        return jsonify({"verified": True, "similarity": float(similarity)})
    return jsonify({"verified": False, "similarity": float(similarity)})


if __name__ == "__main__":
    args = parse_args()
    app.run(host='127.0.0.1', port = args.port)