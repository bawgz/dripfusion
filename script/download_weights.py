# Run this before you deploy it on replicate, because if you don't
# whenever you run the model, it will download the weights from the
# internet, which will take a long time.

import torch
import os
from diffusers import AutoencoderKL, DiffusionPipeline

better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipe = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V3.0",
    vae=better_vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.load_lora_weights("bawgz/dripfusion", weight_name="drip_glasses.safetensors", adapter_name="DRIP", token=os.getenv("token"))

pipe.set_adapters(["DRIP"], adapter_weights=[0.6])

pipe.fuse_lora()

pipe.save_pretrained("./sdxl-cache", safe_serialization=True)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# TODO - we don't need to save all of this and in fact should save just the unet, tokenizer, and config.
pipe.save_pretrained("./refiner-cache", safe_serialization=True)
