from typing import List, Optional
import os
import time
import subprocess

import torch
from safetensors.torch import load_file
from cog import BasePredictor, Input, Path
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    AutoencoderKL,
    PNDMScheduler
)

from dataset_and_utils import TokenEmbeddingsHandler

SDXL_MODEL_CACHE = "./sdxl-cache"

REFINER_MODEL_CACHE = "./refiner-cache"

TRAINED_MODEL_LOCATION = "./trained-model"

SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

SCHEDULERS = {
  "DDIM": DDIMScheduler,
  "DPMSolverMultistep": DPMSolverMultistepScheduler,
  "HeunDiscrete": HeunDiscreteScheduler,
  "KarrasDPM": KarrasDPM,
  "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
  "K_EULER": EulerDiscreteScheduler,
  "PNDM": PNDMScheduler,
}

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        print("weights: ", weights)
        
        if str(weights) == "weights":
            weights = None

        print("Loading sdxl txt2img pipeline...")

        if not os.path.exists(SDXL_MODEL_CACHE):
            better_vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16
            )

            self.pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                vae=better_vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

            self.pipe.save_pretrained("./sdxl-cache", safe_serialization=True)
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                SDXL_MODEL_CACHE,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )

        self.pipe.load_lora_weights("./", weight_name="drip_glasses.safetensors", adapter_name="DRIP")

        self.trained_model = weights or os.path.exists(TRAINED_MODEL_LOCATION)

        if self.trained_model:
            if not os.path.exists(TRAINED_MODEL_LOCATION):
                print("downloading weights")
                download_weights(weights, TRAINED_MODEL_LOCATION)
            
            print("Loading fine-tuned model")
            state_dict = load_file(os.path.join(TRAINED_MODEL_LOCATION, "embeddings.pti"))

            # notice we load the tokens <s0><s1>, as "TOK" as only a place-holder and training was performed using the new initialized tokens - <s0><s1>
            # load embeddings of text_encoder 1 (CLIP ViT-L/14)
            self.pipe.load_textual_inversion(state_dict["text_encoders_0"], token=["<s0>", "<s1>"], text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer)
            # load embeddings of text_encoder 2 (CLIP ViT-G/14)
            self.pipe.load_textual_inversion(state_dict["text_encoders_1"], token=["<s0>", "<s1>"], text_encoder=self.pipe.text_encoder_2, tokenizer=self.pipe.tokenizer_2)
            self.pipe.load_lora_weights(TRAINED_MODEL_LOCATION, weight_name="lora.safetensors", adapter_name="TOK")

        print("Loading SDXL refiner pipeline...")
        # FIXME(ja): should the vae/text_encoder_2 be loaded from SDXL always?
        #            - in the case of fine-tuned SDXL should we still?
        # FIXME(ja): if the answer to above is use VAE/Text_Encoder_2 from fine-tune
        #            what does this imply about lora + refiner? does the refiner need to know about

        print("Loading refiner pipeline...")

        if os.path.exists(REFINER_MODEL_CACHE):
            self.refiner = DiffusionPipeline.from_pretrained(
                REFINER_MODEL_CACHE,
                text_encoder_2=self.pipe.text_encoder_2,
                vae=self.pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )           
        else:
            self.refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=self.pipe.text_encoder_2,
                vae=self.pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

            self.refiner.save_pretrained("./refiner-cache", safe_serialization=True)
        
        self.refiner.to("cuda")


        # FIXME: should I load lora weights to the refiner? Below does not work
        # print("setting refiner adapters")
        # self.refiner.load_lora_weights("./trained-model-luk/", weight_name="lora.safetensors", adapter_name="LUK")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        lora_scale_base: float = Input(
            description="LoRA additive scale for base dripfusion lora",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        lora_scale_custom: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        replicate_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
        custom_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "expert_ensemble_refiner", "base_image_refiner"],
            default="no_refiner",
        ),
        high_noise_frac: float = Input(
            description="For expert_ensemble_refiner, the fraction of noise to use",
            default=0.8,
            le=1.0,
            ge=0.0,
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
            default=None,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model."""

        prompt = prompt.replace("TOK", "<s0><s1>")
        print(f"Prompt: {prompt}")

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if self.replicate_weights:
            print("downloading weights")
            self.trained_model = True
            download_weights(replicate_weights, TRAINED_MODEL_LOCATION)
            
            state_dict = load_file(os.path.join(TRAINED_MODEL_LOCATION, "embeddings.pti"))

            # notice we load the tokens <s0><s1>, as "TOK" as only a place-holder and training was performed using the new initialized tokens - <s0><s1>
            # load embeddings of text_encoder 1 (CLIP ViT-L/14)
            self.pipe.load_textual_inversion(state_dict["text_encoders_0"], token=["<s0>", "<s1>"], text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer)
            # load embeddings of text_encoder 2 (CLIP ViT-G/14)
            self.pipe.load_textual_inversion(state_dict["text_encoders_1"], token=["<s0>", "<s1>"], text_encoder=self.pipe.text_encoder_2, tokenizer=self.pipe.tokenizer_2)
            self.pipe.load_lora_weights(TRAINED_MODEL_LOCATION, weight_name="lora.safetensors", adapter_name="TOK")


        if self.trained_model:
            print("using two loras")
            self.pipe.set_adapters(["TOK", "DRIP"], adapter_weights=[lora_scale_custom, lora_scale_base])

        sdxl_kwargs = {}

        if refine == "expert_ensemble_refiner":
            sdxl_kwargs["output_type"] = "latent"
            sdxl_kwargs["denoising_end"] = high_noise_frac
        elif refine == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"

        sdxl_kwargs["cross_attention_kwargs"] = {"scale": 1.0 if self.trained_model else lora_scale_base}

        common_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": torch.manual_seed(seed),
            "num_inference_steps": num_inference_steps,
        }

        pipe = self.pipe

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)

        print("Common args: ", common_args)
        print("SDXL args: ", sdxl_kwargs)

        output = pipe(**common_args, **sdxl_kwargs)

        if refine in ["expert_ensemble_refiner", "base_image_refiner"]:
            refiner_kwargs = {
                "image": output.images,
            }

            if refine == "expert_ensemble_refiner":
                refiner_kwargs["denoising_start"] = high_noise_frac
            if refine == "base_image_refiner" and refine_steps:
                common_args["num_inference_steps"] = refine_steps

            output = self.refiner(**common_args, **refiner_kwargs)

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths