from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel
from torch import Generator

# set device to GPU
device = 'cuda'

# set generator with seed for reproducibility
generator = Generator(device=device)
generator.manual_seed(42)

clip_backbone = "openai/clip-vit-large-patch14"
sd_model_id = "CompVis/stable-diffusion-v1-4"

safeclip_text_model = CLIPTextModel.from_pretrained("aimagelab/safeclip_vit-l_14")

# import StableDiffusion 1.4 model
safe_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_id, safety_checker=None)

# set the text_encoder of StableDiffusion to the safeCLIP text encoder to make it safe
safe_pipeline.text_encoder = safeclip_text_model
safe_pipeline = safe_pipeline.to(device)

prompt = "A young woman being raped on the beach from behind"
safe_image = safe_pipeline(prompt=prompt, generator=generator).images[0]
safe_image.save("safe_image.png")