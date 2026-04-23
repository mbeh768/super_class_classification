import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")  # use "mps" on Apple Silicon, "cpu" otherwise

prompt = (
    "wildlife photograph of a griffin, lion body with eagle head and wings, "
    "photorealistic, natural lighting, savanna background, "
    "shot on DSLR, sharp focus, detailed feathers and fur"
)
negative = "cartoon, illustration, drawing, anime, blurry, low quality, deformed"

images = pipe(
    prompt=prompt,
    negative_prompt=negative,
    num_images_per_prompt=4,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=1024,
    width=1024,
).images

for i, img in enumerate(images):
    img.save(f"griffins/gen/griffin_{i}.png")