# app.py
import gradio as gr
from detect import detect_gun

iface = gr.Interface(
    fn=detect_gun,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Image(type="numpy", label="Detection Result"),
    title="🚨 Real-Time Gun Detection",
    description="Upload an image to detect firearms using a YOLOv8 model fine-tuned on the Armas dataset."
)

if __name__ == "__main__":
    iface.launch()
