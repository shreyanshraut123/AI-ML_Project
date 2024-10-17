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


# Define a function to return information based on the input
def get_information(term):
    # You can expand this dictionary with more terms and their information
    info_dict = {
        "tiger": "The tiger is the largest species of the cat family, native to Asia. It is known for its strength and beautiful striped coat.",
        "lion": "Lions are social big cats that live in groups called prides. They are known as the 'king of the jungle'.",
        "elephant": "Elephants are the largest land animals on Earth, known for their intelligence, memory, and strong social bonds.",
    }

    return info_dict.get(term.lower(), None)


# Function to generate mobile UI design or provide information
def generate_mobile_ui(prompt):
    # Check if the prompt matches any known terms for information
    information = get_information(prompt)

    if information:
        # Return the information if a known term is matched
        return None, information  # No image, but return the information

    # Generate image based on the input prompt
    image = pipe(prompt).images[0]

    # Usage information about the prompt
    info = f"The input prompt was: '{prompt}'\nThe design generated is based on the description of a mobile UI."

    return image, info  # Return the generated image and usage information


# Create a Gradio interface with both image and text output
interface = gr.Interface(
    fn=generate_mobile_ui,  # Function that generates images or provides info
    inputs="text",  # Input type (text prompt)
    outputs=["image", "text"],  # Output both image and text
    title="Mobile UI Design Generator and Info Lookup",  # Title of the interface
    description="Enter a description of the mobile UI design or a term to get information.",  # Description for users
    examples=[
        # "A modern, minimalist mobile app login screen with light colors",
        "tiger"
        # "A dashboard for a financial app with charts and account summaries"
    ]  # Example prompts for users
)

# Launch the Gradio interface
interface.launch()
