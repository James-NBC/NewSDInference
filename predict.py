# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
import json
import tempfile as tf
from cog import BasePredictor, Input
from model import ImageGenerator, Verifier    

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
        prompt: str = Input(description="Prompt to generate an image from"),
        seed: str = Input(description="Seed for random number generator"),
        path: str = Input(description="Path to save the generated image or to check the image"),
        verify: bool = Input(description="Whether to verify the generated image", default=False)
    ) -> bool:
        """Generate an image from a prompt"""
        int_seed = int(seed[2:], 16) % (2**32)
        checked_image = self.model(prompt, int_seed)
        if not verify:
            checked_image.save(path, optimize=True, quality=40)
        else:
            temp_file_path = tf.mkstemp(suffix= os.path.basename(path))[1]
            checked_image.save(temp_file_path, optimize=True, quality=40)
            return self.verifier(path, temp_file_path)
        return True