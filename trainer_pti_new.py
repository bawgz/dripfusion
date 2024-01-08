# Bootstrapped from Huggingface diffuser's code.
import fnmatch
import json
import math
import os
import shutil
from typing import List, Optional

import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
from torchvision import transforms

from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.optimization import get_scheduler
from safetensors.torch import save_file
from tqdm.auto import tqdm
from peft import LoraConfig

from dataset_and_utils import (
    PreprocessedDataset,
    TokenEmbeddingsHandler,
    load_models,
    unet_attn_processors_state_dict,
)


def main(
    pretrained_model_name_or_path: Optional[
        str
    ] = "./cache",  # "stabilityai/stable-diffusion-xl-base-1.0",
    revision: Optional[str] = None,
    instance_data_dir: Optional[str] = "./dataset/zeke/captions.csv",
    output_dir: str = "ft_masked_coke",
    seed: Optional[int] = 42,
    resolution: int = 512,
    crops_coords_top_left_h: int = 0,
    crops_coords_top_left_w: int = 0,
    train_batch_size: int = 1,
    do_cache: bool = True,
    num_train_epochs: int = 600,
    max_train_steps: Optional[int] = None,
    checkpointing_steps: int = 500000,  # default to no checkpoints
    gradient_accumulation_steps: int = 1,  # todo
    lora_lr: float = 1e-4,
    pivot_halfway: bool = True,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 500,
    lr_num_cycles: int = 1,
    lr_power: float = 1.0,
    dataloader_num_workers: int = 0,
    max_grad_norm: float = 1.0,  # todo with tests
    allow_tf32: bool = True,
    mixed_precision: Optional[str] = "bf16",
    device: str = "cuda:0",
    verbose: bool = True,
    is_lora: bool = True,
    lora_rank: int = 4,
    num_processes: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
) -> None:
    # If passed along, set the training seed now.
    if not seed:
        seed = np.random.randint(0, 2**32 - 1)
    print("Using seed", seed)
    torch.manual_seed(seed)

    # Handle the repository creation
    os.makedirs(output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if mixed_precision == "fp16":
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    # TODO: maybe worth spending time figuring this jawn out
    # if enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warn(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    # TODO: what this is?
    # if gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if scale_lr:
        lora_lr = (
            lora_lr * gradient_accumulation_steps * train_batch_size * num_processes
        )

    # TODO: what this is?
    # Initialize the optimizer
    # if use_8bit_adam:
    #     try:
    #         import bitsandbytes as bnb
    #     except ImportError:
    #         raise ImportError(
    #             "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
    #         )

    #     optimizer_cls = bnb.optim.AdamW8bit
    # else:
    #     optimizer_cls = torch.optim.AdamW

    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=lora_lr,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # from old Replicate way... just here for reference
    # train_dataset = PreprocessedDataset(
    #     instance_data_dir,
    #     tokenizer_one,
    #     tokenizer_two,
    #     vae.float(),
    #     do_cache=True,
    #     substitute_caption_map=token_dict,
    # )
    print(instance_data_dir)
    data_files = {}
    data_files["train"] = os.path.join(instance_data_dir, "**")
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        # cache_dir=cache_dir,
    )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    print("dataset", dataset)
    print("column_names", column_names)

    # I can do this much simpler than the original code
    # 6. Get the column names for input/target.
    # dataset_columns = DATASET_NAME_MAPPING.get(dataset_name, None)
    # if image_column is None:
    #     image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    # else:
    #     image_column = image_column
    #     if image_column not in column_names:
    #         raise ValueError(
    #             f"--image_column' value '{image_column}' needs to be one of: {', '.join(column_names)}"
    #         )
    # if caption_column is None:
    #     caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    # else:
    #     caption_column = caption_column
    #     if caption_column not in column_names:
    #         raise ValueError(
    #             f"--caption_column' value '{caption_column}' needs to be one of: {', '.join(column_names)}"
    #         )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples):
        captions = []
        for caption in examples["caption"]:
            captions.append(caption)

        print("captions ", captions)

        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # def tokenize_captions(dataset, is_train=True):
    #     captions = []
    #     for caption in dataset["captions"]:
    #         if isinstance(caption, str):
    #             captions.append(caption)
    #         elif isinstance(caption, (list, np.ndarray)):
    #             # take a random caption if there are multiple
    #             captions.append(random.choice(caption) if is_train else caption[0])
    #         else:
    #             raise ValueError(
    #                 f"Caption column `{caption_column}` should contain either strings or lists of strings."
    #             )
    #     inputs = tokenizer(
    #         captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    #     )
    #     return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * num_processes,
        num_training_steps=max_train_steps * num_processes,
    )

    print("train_dataset", train_dataset)
    print("lr_scheduler", lr_scheduler)

    # # Prepare everything with our `accelerator`.
    # unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     unet, optimizer, train_dataloader, lr_scheduler
    # )

    # # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # if overrode_max_train_steps:
    #     max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    # num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # # We need to initialize the trackers we use, and also store our configuration.
    # # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # # Train!
    # total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num Epochs = {num_train_epochs}")
    # logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    # logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    # logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {max_train_steps}")
    # global_step = 0
    # first_epoch = 0

    # # Potentially load in the weights and states from a previous save
    # if resume_from_checkpoint:
    #     if resume_from_checkpoint != "latest":
    #         path = os.path.basename(resume_from_checkpoint)
    #     else:
    #         # Get the most recent checkpoint
    #         dirs = os.listdir(output_dir)
    #         dirs = [d for d in dirs if d.startswith("checkpoint")]
    #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    #         path = dirs[-1] if len(dirs) > 0 else None

    #     if path is None:
    #         accelerator.print(
    #             f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
    #         )
    #         resume_from_checkpoint = None
    #         initial_global_step = 0
    #     else:
    #         accelerator.print(f"Resuming from checkpoint {path}")
    #         accelerator.load_state(os.path.join(output_dir, path))
    #         global_step = int(path.split("-")[1])

    #         initial_global_step = global_step
    #         first_epoch = global_step // num_update_steps_per_epoch
    # else:
    #     initial_global_step = 0

    # progress_bar = tqdm(
    #     range(0, max_train_steps),
    #     initial=initial_global_step,
    #     desc="Steps",
    #     # Only show the progress bar once on each machine.
    #     disable=not accelerator.is_local_main_process,
    # )

    # for epoch in range(first_epoch, num_train_epochs):
    #     unet.train()
    #     train_loss = 0.0
    #     for step, batch in enumerate(train_dataloader):
    #         with accelerator.accumulate(unet):
    #             # Convert images to latent space
    #             latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
    #             latents = latents * vae.config.scaling_factor

    #             # Sample noise that we'll add to the latents
    #             noise = torch.randn_like(latents)
    #             if noise_offset:
    #                 # https://www.crosslabs.org//blog/diffusion-with-offset-noise
    #                 noise += noise_offset * torch.randn(
    #                     (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
    #                 )

    #             bsz = latents.shape[0]
    #             # Sample a random timestep for each image
    #             timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    #             timesteps = timesteps.long()

    #             # Add noise to the latents according to the noise magnitude at each timestep
    #             # (this is the forward diffusion process)
    #             noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    #             # Get the text embedding for conditioning
    #             encoder_hidden_states = text_encoder(batch["input_ids"])[0]

    #             # Get the target for loss depending on the prediction type
    #             if prediction_type is not None:
    #                 # set prediction_type of scheduler if defined
    #                 noise_scheduler.register_to_config(prediction_type=prediction_type)

    #             if noise_scheduler.config.prediction_type == "epsilon":
    #                 target = noise
    #             elif noise_scheduler.config.prediction_type == "v_prediction":
    #                 target = noise_scheduler.get_velocity(latents, noise, timesteps)
    #             else:
    #                 raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    #             # Predict the noise residual and compute loss
    #             model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    #             if snr_gamma is None:
    #                 loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    #             else:
    #                 # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
    #                 # Since we predict the noise instead of x_0, the original formulation is slightly changed.
    #                 # This is discussed in Section 4.2 of the same paper.
    #                 snr = compute_snr(noise_scheduler, timesteps)
    #                 if noise_scheduler.config.prediction_type == "v_prediction":
    #                     # Velocity objective requires that we add one to SNR values before we divide by them.
    #                     snr = snr + 1
    #                 mse_loss_weights = (
    #                     torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
    #                 )

    #                 loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    #                 loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    #                 loss = loss.mean()

    #             # Gather the losses across all processes for logging (if we use distributed training).
    #             avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
    #             train_loss += avg_loss.item() / gradient_accumulation_steps

    #             # Backpropagate
    #             accelerator.backward(loss)
    #             if accelerator.sync_gradients:
    #                 params_to_clip = lora_layers
    #                 accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()

    #         # Checks if the accelerator has performed an optimization step behind the scenes
    #         if accelerator.sync_gradients:
    #             progress_bar.update(1)
    #             global_step += 1
    #             accelerator.log({"train_loss": train_loss}, step=global_step)
    #             train_loss = 0.0

    #             if global_step % checkpointing_steps == 0:
    #                 if accelerator.is_main_process:
    #                     # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    #                     if checkpoints_total_limit is not None:
    #                         checkpoints = os.listdir(output_dir)
    #                         checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    #                         checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    #                         # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    #                         if len(checkpoints) >= checkpoints_total_limit:
    #                             num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
    #                             removing_checkpoints = checkpoints[0:num_to_remove]

    #                             logger.info(
    #                                 f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
    #                             )
    #                             logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

    #                             for removing_checkpoint in removing_checkpoints:
    #                                 removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
    #                                 shutil.rmtree(removing_checkpoint)

    #                     save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    #                     accelerator.save_state(save_path)

    #                     unwrapped_unet = accelerator.unwrap_model(unet)
    #                     unet_lora_state_dict = convert_state_dict_to_diffusers(
    #                         get_peft_model_state_dict(unwrapped_unet)
    #                     )

    #                     StableDiffusionPipeline.save_lora_weights(
    #                         save_directory=save_path,
    #                         unet_lora_layers=unet_lora_state_dict,
    #                         safe_serialization=True,
    #                     )

    #                     logger.info(f"Saved state to {save_path}")

    #         logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
    #         progress_bar.set_postfix(**logs)

    #         if global_step >= max_train_steps:
    #             break

    #     if accelerator.is_main_process:
    #         if validation_prompt is not None and epoch % validation_epochs == 0:
    #             logger.info(
    #                 f"Running validation... \n Generating {num_validation_images} images with prompt:"
    #                 f" {validation_prompt}."
    #             )
    #             # create pipeline
    #             pipeline = DiffusionPipeline.from_pretrained(
    #                 pretrained_model_name_or_path,
    #                 unet=accelerator.unwrap_model(unet),
    #                 revision=revision,
    #                 variant=variant,
    #                 torch_dtype=weight_dtype,
    #             )
    #             pipeline = pipeline.to(accelerator.device)
    #             pipeline.set_progress_bar_config(disable=True)

    #             # run inference
    #             generator = torch.Generator(device=accelerator.device)
    #             if seed is not None:
    #                 generator = generator.manual_seed(seed)
    #             images = []
    #             with torch.cuda.amp.autocast():
    #                 for _ in range(num_validation_images):
    #                     images.append(
    #                         pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0]
    #                     )

    #             for tracker in accelerator.trackers:
    #                 if tracker.name == "tensorboard":
    #                     np_images = np.stack([np.asarray(img) for img in images])
    #                     tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
    #                 if tracker.name == "wandb":
    #                     tracker.log(
    #                         {
    #                             "validation": [
    #                                 wandb.Image(image, caption=f"{i}: {validation_prompt}")
    #                                 for i, image in enumerate(images)
    #                             ]
    #                         }
    #                     )

    #             del pipeline
    #             torch.cuda.empty_cache()

    # # Save the lora layers
    # accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     unet = unet.to(torch.float32)

    #     unwrapped_unet = accelerator.unwrap_model(unet)
    #     unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
    #     StableDiffusionPipeline.save_lora_weights(
    #         save_directory=output_dir,
    #         unet_lora_layers=unet_lora_state_dict,
    #         safe_serialization=True,
    #     )

    #     if push_to_hub:
    #         save_model_card(
    #             repo_id,
    #             images=images,
    #             base_model=pretrained_model_name_or_path,
    #             dataset_name=dataset_name,
    #             repo_folder=output_dir,
    #         )
    #         upload_folder(
    #             repo_id=repo_id,
    #             folder_path=output_dir,
    #             commit_message="End of training",
    #             ignore_patterns=["step_*", "epoch_*"],
    #         )

    #     # Final inference
    #     # Load previous pipeline
    #     if validation_prompt is not None:
    #         pipeline = DiffusionPipeline.from_pretrained(
    #             pretrained_model_name_or_path,
    #             revision=revision,
    #             variant=variant,
    #             torch_dtype=weight_dtype,
    #         )
    #         pipeline = pipeline.to(accelerator.device)

    #         # load attention processors
    #         pipeline.load_lora_weights(output_dir)

    #         # run inference
    #         generator = torch.Generator(device=accelerator.device)
    #         if seed is not None:
    #             generator = generator.manual_seed(seed)
    #         images = []
    #         with torch.cuda.amp.autocast():
    #             for _ in range(num_validation_images):
    #                 images.append(
    #                     pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0]
    #                 )

    #         for tracker in accelerator.trackers:
    #             if len(images) != 0:
    #                 if tracker.name == "tensorboard":
    #                     np_images = np.stack([np.asarray(img) for img in images])
    #                     tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
    #                 if tracker.name == "wandb":
    #                     tracker.log(
    #                         {
    #                             "test": [
    #                                 wandb.Image(image, caption=f"{i}: {validation_prompt}")
    #                                 for i, image in enumerate(images)
    #                             ]
    #                         }
    #                     )


if __name__ == "__main__":
    main()
