from torchvision import transforms
from PIL import Image
import torch
import base64
from io import BytesIO

# Preprocessing pipeline for CIFAR-10 images
def preprocess_image(base64_string: str) -> torch.Tensor:
    try:
        # Decode base64 to image
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Define the transform pipeline
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

        # Apply transforms
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {e}")
