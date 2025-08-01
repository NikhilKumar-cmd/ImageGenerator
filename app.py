import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_auth_token=hf_token,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Gradio UI
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, label="Enter your prompt"),
    outputs="image",
    title="🖼️ AI Text to Image Generator",
    description="Enter a prompt and get a generated image using Stable Diffusion."
)

if __name__ == "__main__":
    interface.launch(share=True)
