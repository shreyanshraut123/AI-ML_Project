import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
print(torch.cuda.is_available())

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cpu")


# Simple keyword-based retrieval function
def retrieve_info(query):
    info_dict = {
        "login": "A modern, minimalist mobile app login screen with light colors.",
        "dashboard": "A sleek mobile app dashboard UI with dark mode, icons for notifications.",
        "e-commerce": "A mobile e-commerce app with product images and add-to-cart buttons.",
        "profile": "A user profile screen showing personal information and settings.",
        "settings": "A settings screen with toggle switches and sliders."
    }

    for keyword in info_dict.keys():
        if keyword in query.lower():
            return info_dict[keyword]

    return None


# Function for command-line interface
def main():
    print("Welcome to the Mobile UI Design Generator!")
    print("Type 'exit' to quit.")

    while True:
        query = input("Enter your design query: ")
        if query.lower() == 'exit':
            break

        prompt = retrieve_info(query)

        if prompt:
            print(f"Using prompt: '{prompt}'")
            image = pipe(prompt).images[0]
            image.save("generated_mobile_ui.jpg")
            print("Generated mobile UI design saved as 'generated_mobile_ui.jpg'.")
        else:
            print("No matching design prompt found. Please try a different query.")


if __name__ == "__main__":
    main()


# Optional: Gradio interface
def gradio_interface(query):
    prompt = retrieve_info(query)
    if prompt:
        image = pipe(prompt).images[0]
        return image
    else:
        return "No matching design prompt found."


interface = gr.Interface(fn=gradio_interface, inputs="text", outputs="image", title="Mobile UI Design Generator")
interface.launch()
