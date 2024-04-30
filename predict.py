# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
import time
import json
from cog import BaseModel, BasePredictor, Input
from model import ImageGenerator, Verifier


class PredictorOutput(BaseModel):
    output_path: str = ""
    verify: bool = False
    similarity: float = 0.0
    time: float = 0.0


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        assert os.path.exists('config.json'), "config.json not found"
        with open('config.json', 'r') as f:
            config = json.load(f)
        self.model = ImageGenerator(config["stable_diffusion_cfg"])
        self.verifier = Verifier(config["verifier_cfg"])

    def predict(
        self,
        prompt: str = Input(
            description="Prompt to generate an image from"),
        seed: str = Input(
            description="Seed for random number generator"),
            path: str = Input(
                description="Path to save the generated image or to check the image"),
        verify: bool = Input(
            description="Whether to verify the generated image",
            default=False)) -> PredictorOutput:
        start = time.time()
        int_seed = int(seed[2:], 16) % (2**32)
        checked_image = self.model(prompt, int_seed)
        if verify:
            is_same, o_similarity = self.verifier(checked_image, path)
            total_time = time.time() - start
            return PredictorOutput(
                verify=is_same,
                similarity=o_similarity,
                time=total_time)
        checked_image.save(path, optimize=True, quality=40)
        total_time = time.time() - start
        return PredictorOutput(output_path=path, time=total_time)
