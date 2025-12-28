import os

import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

from model import ParkingCNN


WEIGHTS_PATH = os.getenv("PARKING_MODEL_PATH", "parking_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_model(weights_path: str = WEIGHTS_PATH):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. "
            "Place parking_model.pth in the project root or set PARKING_MODEL_PATH."
        )
    model = ParkingCNN(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


MODEL = load_model()


def predict(image: Image.Image):
    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(tensor)
        predicted = torch.argmax(outputs, dim=1).item()

    if predicted == 0:
        return "Empty (Boş)"
    return "Occupied (Dolu)"


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Parking Spot Occupancy Detection",
    description="Upload a parking spot image to classify it as Empty or Occupied.",
)

if __name__ == "__main__":
    interface.launch()
