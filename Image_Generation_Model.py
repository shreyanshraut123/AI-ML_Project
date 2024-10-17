from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cuda")  # If you have a GPU

prompt = "A modern mobile UI login screen with minimalist design"
image = pipe(prompt).images[0]
image.show()
