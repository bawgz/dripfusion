sudo cog predict -i prompt="a photo of TOK man wearing reflective lens sunglasses, instagram" -i negative_prompt="((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (fused fingers), (too many fingers), (((long neck)))" -i lora_scale_base=1.0 -i lora_scale_custom=1.0 -i refine=expert_ensemble_refiner -i high_noise_frac=0.95 -i custom_weights="https://replicate.delivery/pbxt/TnLzbfHJzjSpDiELcVXr377q6M0xWFc8rFwSAQAIcE0xwvFJA/trained_model.tar"

sudo cog predict -i prompt="a photo of TOK man wearing pit viper sunglasses"