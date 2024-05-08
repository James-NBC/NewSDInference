# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
import json
import time
from model import ImageGenerator
from cog import BasePredictor, Input, 

class PredictorOutput(BaseModel):
    result: str
    inference_time: float

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        assert os.path.exists('config.json'), "config.json not found"
        with open('config.json', 'r') as f:
            config = json.load(f)
        self.model = ImageGenerator(config)

    def predict(
        self,
        prompt: str = Input(
            description="Prompt to generate an image from"),
        seed: int = Input(
            description="Seed for random number generator",
            default=0),
        h: int = Input(
            description="Height of the image",
            default=None),
        w: int = Input(
            description="Width of the image",
            default=None),
        steps: int = Input(
            description="Number of inference steps",
            default=None),
        cfg: float = Input(
            description="Guidance scale",
            default=None),
        output_path: str = Input(
            description="Path to save the generated image or to check the image",
            default="output.jpg")
    ) -> PredictorOutput:
        start = time.time()
        checked_image = self.model(prompt, seed, h, w, steps, cfg)
        buffered = BytesIO()
        checked_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()
        return PredictorOutput(result=base64_image, inference_time= time.time() - start)
