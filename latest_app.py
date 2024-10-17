from diffusers import StableDiffusionPipeline
import torch

print(torch.cuda.is_available())  # Should return False since you are using CPU

model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id)  # Remove torch_dtype argument
pipe = pipe.to("cpu")  # Using CPU

prompts = [
    "Alice in Wonderland, Ultra HD, realistic, futuristic, detailed, octane render, photoshopped, photorealistic, soft, pastel, Aesthetic, Magical background",
    "Anime style Alice in Wonderland, 90's vintage style, digital art, ultra HD, 8k, photoshopped, sharp focus, surrealism, akira style, detailed line art",
    # "Beautiful, abstract art of Chesire cat of Alice in wonderland, 3D, highly detailed, 8K, aesthetic"
]

images = []
for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f'picture_{i}.jpg')
    images.append(image)
