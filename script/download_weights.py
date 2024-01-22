# Run this before you deploy it on replicate, because if you don't
# whenever you run the model, it will download the weights from the
# internet, which will take a long time.

import torch
import sys
from diffusers import DiffusionPipeline

def main(token):
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