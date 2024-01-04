---
tags:
  - text-to-image
  - stable-diffusion
  - lora
  - diffusers
  - template:sd-lora
widget:
- text: >-
    a photo of  a person wearing reflective lens sunglasses
    <lora:dripglasses-reflective_lens_sunglasses-000001:1>
  parameters:
    negative_prompt: >-
      ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame],
      extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn
      face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad
      proportions))), ((extra limbs)), cloned face, (((disfigured))), gross
      proportions, (malformed limbs), ((missing arms)), ((missing legs)),
      (((extra arms))), (((extra legs))), (fused fingers), (too many fingers),
      (((long neck)))
  output:
    url: images/00023-23.png
base_model: stabilityai/stable-diffusion-xl-base-1.0
instance_prompt: reflective lens sunglasses

---
# drip-glasses

<Gallery />


## Trigger words

You should use `reflective lens sunglasses` to trigger the image generation.


## Download model

Weights for this model are available in Safetensors format.

[Download](/bawgz/drip-glasses/tree/main) them in the Files & versions tab.
