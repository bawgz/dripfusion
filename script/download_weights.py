# Run this before you deploy it on replicate, because if you don't
# whenever you run the model, it will download the weights from the
# internet, which will take a long time.

import torch
import sys
from diffusers import AutoencoderKL, DiffusionPipeline
from huggingface_hub import hf_hub_download

def main(token):
    hf_hub_download(repo_id="bawgz/dripfusion", filename="drip_glasses.safetensors", repo_type="model", local_dir="./", local_dir_use_symlinks=False, use_auth_token=token)

if __name__ == "__main__":
    hf_token = sys.argv[1]
    main(hf_token)