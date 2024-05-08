import os
import gc
import time
import json
import torch
import base64
from io import BytesIO
from ray import serve
from model import ImageGenerator
from starlette.requests import Request

@serve.deployment
class RayImageGenerator:
    def __init__(self, config):
        self.config = config

    async def __call__(self, http_request: Request) -> str:
        model = ImageGenerator(self.config)
        start = time.time()
        json_request = await http_request.json()
        prompt = json_request["prompt"]
        seed = json_request["seed"]
        h = json_request.get("height", None)
        w = json_request.get("width", None)
        steps = json_request.get("steps", None)
        cfg = json_request.get("cfg", None)
        generated_image = model(prompt, seed, h, w, steps, cfg)
        buffered = BytesIO()
        generated_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()
        del model
        torch.cuda.empty_cache() 
        gc.collect()
        return {"image": base64_image, "inference_time": time.time() - start}

assert os.path.exists("config.json"), "config.json not found"
with open("config.json", "r") as f:
    config = json.load(f)
image_generator_app = RayImageGenerator.bind(config)