# app/vision_classifier.py
import torch
from torchvision import transforms
from PIL import Image
from models.vision.simple_cnn import SimpleCNN  # import model class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = 4
MODEL_PATH = "models/vision/doc_classifier.pth"
classes = ["invoice", "id_card", "certificate", "resume"]

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def classify_image(image: Image.Image) -> str:
    """
    Classify document image into a type
    """
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
    return classes[pred.item()]
