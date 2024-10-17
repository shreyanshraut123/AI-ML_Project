import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Check if CUDA is available for faster image generation
print(torch.cuda.is_available())

# Load the Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Use CPU if no GPU is available
pipe = pipe.to("cpu")

# Function to generate mobile UI designs from text input
def generate_mobile_ui(prompt):
    # Generate image based on the input prompt
    image = pipe(prompt).images[0]
    return image

# Create a Gradio interface
interface = gr.Interface(
    fn=generate_mobile_ui,  # Function that generates images
    inputs="text",  # Input type (text prompt)
    outputs="image",  # Output type (generated image)
    title="Mobile UI Design Generator",  # Title of the interface
    description="Enter a description of the mobile UI design you'd like to generate.",  # Description for users
    examples=[
        "A mobile e-commerce app with product images and add-to-cart buttons"
    ]  # Example prompts for users
)

# Launch the Gradio interface
interface.launch()
