# Author: Gentry Atkinson
# Organization: St. Edwards University
# Date: 21 Mar, 2024

# Make attractive promotional images for the podcast
# Ideal cover art size -> 3000 x 3000 px as a png or jpeg

import os
import torch
from diffusers import DiffusionPipeline

def gen_cover_art(device: str):
    model = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(
        model, 
        torch_dtype=torch.float16, use_safetensors=True, 
        variant="fp16"
    )
    pipe = pipe.to(device)

    prompt = "Generate cover art for a science-focused podcast called No Humans Were Involved with this Podcast. The cover art should be bright and eye catching."
    
    image = pipe(prompt=prompt).images[0]
    image = image.resize((1400, 1400))
    image.save(os.path.join("Robot-Podcast-Generator", "imgs", "cover.png"))

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    gen_cover_art(device)
