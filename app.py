import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

from model import ParkingCNN


def load_model(weights_path: str = "parking_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParkingCNN(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict(image: Image.Image):
    model, device = load_model()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
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
