import os
import json
from ray import serve
from model import ImageGenerator
from starlette.requests import Request

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.2})
class RayImageGenerator:
    def __init__(self):
        assert os.path.exists("config.json"), "config.json not found"
        with open("config.json", "r") as f:
            config = json.load(f)
        self.model = ImageGenerator(config)

    async def __call__(self, http_request: Request) -> str:
        json_request = await http_request.json()
        prompt = json_request["prompt"]
        seed = json_request["seed"]
        output_path = json_request.get("output_path", "output.jpg")
        h = json_request.get("height", None)
        w = json_request.get("width", None)
        steps = json_request.get("steps", None)
        cfg = json_request.get("cfg", None)
        generated_image = self.model(prompt, seed, h, w, steps, cfg)
        generated_image.save(output_path, optimized = True, quality = 40)
        return output_path

image_generator_app = RayImageGenerator.bind()