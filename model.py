import os
import torch
import diffusers
import numpy as np
from PIL import Image 
import tempfile as tf
from open_clip.model import build_model_from_openai_state_dict
from open_clip.transform import PreprocessCfg, image_transform_v2
from transformers import CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

class Verifier:
    def __init__(self, config, device = None):
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.verifier = torch.jit.load(self.config["ckpt_path"], map_location="cpu").eval()
        self.verifier = build_model_from_openai_state_dict(self.verifier.state_dict(), cast_dtype = torch.float16).to(self.device)
        self.preprocess = image_transform_v2(
            PreprocessCfg(**self.config["preprocess_cfg"]),
            is_train = False
        )
        self.threshold = self.config["threshold"]

    def __call__(self, image, verify_image_path):
        temp_file_path = tf.mkstemp(suffix= os.path.basename(verify_image_path))[1]
        image.save(temp_file_path, optimize=True, quality=40)
        image1 = Image.open(temp_file_path).convert("RGB")
        image2 = Image.open(verify_image_path).convert("RGB")
        image1 = self.preprocess(image1).unsqueeze(0).to(self.device)
        image2 = self.preprocess(image2).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features_1 = self.verifier.encode_image(image1)
            image_features_2 = self.verifier.encode_image(image2)
            image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
            image_features_2 /= image_features_2.norm(dim=-1, keepdim=True)
            similarity = (image_features_1 @ image_features_2.T).mean()
            similarity = min(float(similarity.item()), 1.0)
        torch.cuda.empty_cache()
        return ((similarity >= self.threshold), similarity)


class ImageGenerator:
    def __init__(self, config, device = None):
        self.device = device 
        if self.device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.pipeline = self._load_pipeline()
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(self.config["safety_checker_ckpt"]).to(self.device)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(self.config["feature_extractor_ckpt"])
        self.generator = torch.Generator(device=self.device)

    def _load_pipeline(self):
        ckpt_type = self.config["checkpoint_type"]
        ckpt_dir_or_path = self.config["stable_diffusion_ckpt"]
        if ckpt_type == "safetensors":    
            pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(ckpt_dir_or_path).to(self.device)
        elif ckpt_type == "lora":
            base_model = self.config["base_model"]
            if base_model == "SDXL1.0":
                pipeline = diffusers.StableDiffusionXLPipeline.from_single_file("/home/xxx/base_model/sdxl_1.safetensors").to("cuda")
                pipeline.load_lora_weights(ckpt_dir_or_path)
            else:
                raise ValueError(f"Unknown base model: {base_model}")
        elif ckpt_type == "diffusers":
            pipeline = diffusers.StableDiffusionPipeline.from_pretrained(ckpt_dir_or_path).to(self.device)
        else:
            raise ValueError(f"Unknown checkpoint type: {ckpt_type}")
        return pipeline

    def _check_nsfw_images(self, images):
        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(self.device)
        images_np = [np.array(img) for img in images]

        _, has_nsfw_concepts = self.safety_checker(
            images=images_np,
            clip_input=safety_checker_input.pixel_values.to(self.device),
        )
        return has_nsfw_concepts

    def __call__(self, prompt, seed):
        self.generator.manual_seed(seed)
        image_latents = torch.randn(
            (1, self.pipeline.unet.config.in_channels, self.config["height"] // 8, self.config["width"] // 8),
            generator = self.generator,
            device = self.device
        )
        pil_images = self.pipeline(
            prompt,
            latents = image_latents,
            height=self.config["height"],
            width=self.config["width"],
            guidance_scale=self.config["cfg"],
            num_images_per_prompt=1,
            num_inference_steps=self.config["diffusion_steps"],
        ).images
        has_nsfw_concepts = self._check_nsfw_images(pil_images)
        checked_image = pil_images[0]
        if has_nsfw_concepts[0]:
            checked_image = Image.new("RGB", (self.config["height"], self.config["width"]), (0, 0, 0))
        torch.cuda.empty_cache()
        return checked_image