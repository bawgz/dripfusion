# Run this before you deploy it on replicate, because if you don't
# whenever you run the model, it will download the weights from the
# internet, which will take a long time.

import torch
import sys
from diffusers import DiffusionPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download

def main(token):
    # hf_hub_download(repo_id="bawgz/dripglasses_lora", filename="pit_viper_sunglasses.safetensors", repo_type="model", local_dir="./", local_dir_use_symlinks=False, use_auth_token=token)

    better_vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=better_vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe.save_pretrained("./sdxl-cache")

    dripfusion_pipe = DiffusionPipeline.from_pretrained(
        "bawgz/dripfusion-base",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    dripfusion_pipe.save_pretrained("./dripfusion-cache")

if __name__ == "__main__":
    hf_token = sys.argv[1]
    main(hf_token)