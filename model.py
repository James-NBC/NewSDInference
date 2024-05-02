import os
import torch
import shutil
import diffusers

class ImageGenerator:
    def __init__(self, config, device=None):
        self.device = device
        if self.device is None:
            self.device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.pipeline = self._load_pipeline()
        self.generator = torch.Generator(device=self.device)

    def _load_pipeline(self):
        ckpt_type = self.config["checkpoint_type"]
        if ckpt_type == "safetensors":
            pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(
                self.config["stable_diffusion_ckpt"]).to(self.device)
        elif ckpt_type == "lora":
            base_model = self.config["base_model"]
            base_model_ckpt = os.path.join("checkpoints", base_model + ".safetensors")
            if base_model == "SDXL1.0":
                shutil.copy("/home/xxx/base_model/sdxl_1.safetensors", base_model_ckpt)
                pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(
                    base_model_ckpt).to(self.device)
            else:
                shutil.copy("/home/xxx/base_model/sd_15.safetensors", base_model_ckpt)
                pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
                    base_model_ckpt).to(self.device)
            pipeline.load_lora_weights(self.config["stable_diffusion_ckpt"])
        elif ckpt_type == "diffusers":
            pipeline = diffusers.StableDiffusionPipeline.from_pretrained(
                self.config["stable_diffusion_ckpt"]).to(self.device)
        else:
            raise ValueError(f"Unknown checkpoint type: {ckpt_type}")
        return pipeline

    def __call__(self, prompt, seed):
        self.generator.manual_seed(seed)
        image_latents = torch.randn(
            (1,
             self.pipeline.unet.config.in_channels,
             self.config["height"] // 8,
             self.config["width"] // 8),
            generator=self.generator,
            device=self.device)
        pil_images = self.pipeline(
            prompt,
            latents=image_latents,
            height=self.config["height"],
            width=self.config["width"],
            guidance_scale=self.config["cfg"],
            num_images_per_prompt=1,
            num_inference_steps=self.config["diffusion_steps"],
        ).images
        checked_image = pil_images[0]
        torch.cuda.empty_cache()
        return checked_image