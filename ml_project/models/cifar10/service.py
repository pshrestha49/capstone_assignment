from fastapi import APIRouter, UploadFile, File
from models.cifar10.archi import CifarCNN
import torch
from torchvision import transforms
from PIL import Image

router = APIRouter()

model = CifarCNN()
model.load_state_dict(torch.load("models/cifar10/cifar10_cnn.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # shape: (1, 3, 32, 32)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class_idx = torch.argmax(output, dim=1).item()
        predicted_class_name = CIFAR10_CLASSES[predicted_class_idx]
    return {
        "prediction_index": predicted_class_idx,
        "prediction_label": predicted_class_name
    }
