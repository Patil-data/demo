import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Load TensorFlow model
def load_tensorflow_model(model_path="models/model.h5"):
    try:
        model = tf.keras.models.load_model(model_path)
        print("TensorFlow model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading TensorFlow model: {e}")
        return None

# Load PyTorch model
class DiseaseClassifier(nn.Module):
    def __init__(self):
        super(DiseaseClassifier, self).__init__()
        self.fc = nn.Linear(1024, 1)  # Adjust layers based on your model architecture

    def forward(self, x):
        return self.fc(x)

def load_pytorch_model(model_path="models/model.pth"):
    try:
        model = DiseaseClassifier()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("PyTorch model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None

# Preprocess X-ray image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = Image.open(image)
    return transform(img).unsqueeze(0)

# Make predictions
def predict_disease_tensorflow(model, image):
    try:
        img_array = np.array(Image.open(image).resize((224, 224))).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        return "Pneumonia" if prediction[0][0] > 0.5 else "Healthy"
    except Exception as e:
        print(f"Error making TensorFlow prediction: {e}")
        return "Error"

def predict_disease_pytorch(model, image):
    try:
        img_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(img_tensor)
        return "Pneumonia" if output.item() > 0.5 else "Healthy"
    except Exception as e:
        print(f"Error making PyTorch prediction: {e}")
        return "Error"
