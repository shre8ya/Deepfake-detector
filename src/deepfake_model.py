import torch
import cv2
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Load pretrained EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 1)  # binary output

# ⚠️ We will load weights later
try:
    model.load_state_dict(torch.load("deepfake_weights.pth", map_location="cpu"))
except:
    print("⚠️ No trained weights found. Using random weights.")

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def predict_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return 0.5

    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]

    if face.size == 0:
        return 0.5

    image = Image.fromarray(face)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    score = torch.sigmoid(output)[0].item()

    return score