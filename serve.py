import os
import time
import json
import torch
import loguru
import base64
from io import BytesIO
from ray import serve
from model import ImageGenerator
from starlette.requests import Request

@serve.deployment
class RayImageGenerator:
    def __init__(self, config):
        loguru.logger.info("Loading model")
        self.config = config
        self.model = ImageGenerator(self.config)
        loguru.logger.info("Model loaded")

    async def __call__(self, http_request: Request) -> str:
        start = time.time()
        json_request = await http_request.json()
        prompt = json_request["prompt"]
        seed = json_request.get("seed", None)
        h = json_request.get("height", None)
        w = json_request.get("width", None)
        steps = json_request.get("steps", None)
        cfg = json_request.get("cfg", None)
        loguru.logger.info(f"Run prediction with prompt: {prompt}")
        generated_image = self.model(prompt, seed, h, w, steps, cfg)
        buffered = BytesIO()
        generated_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()
        torch.cuda.empty_cache() 
        infer_time = time.time() - start
        loguru.logger.info(f"Prediction completed in {infer_time} seconds")
        return {"result": base64_image, "inference_time": infer_time}

assert os.path.exists("config.json"), "config.json not found"
with open("config.json", "r") as f:
    config = json.load(f)
app = RayImageGenerator.bind(config)