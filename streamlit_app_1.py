import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

# Page config
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")
st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image to classify it using transfer learning (MobileNetV2).")

# CIFAR-10 Class labels
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

@st.cache_resource
def load_model():
    # Instantiate base architecture
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    
    # Load state dict
    model.load_state_dict(torch.load('mobilenet_cifar10.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_model()
    st.success("Model 'mobilenet_cifar10.pth' loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Transform image to match CIFAR-10 input pipeline
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    input_tensor = transform(image).unsqueeze(0)

    if st.button('Classify Image'):
        with st.spinner('Classifying...'):
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                label = CLASSES[predicted_idx.item()]
                score = confidence.item() * 100

        st.markdown(f"### Prediction: **{label.upper()}**")
        st.write(f"Confidence: **{score:.2f}%**")
