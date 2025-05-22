import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb protection

# Cache model loading
@st.cache_resource
def load_model():
    model = torchvision.models.resnet18(weights="DEFAULT")
    model.eval()
    
    # Set up activation hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.avgpool.register_forward_hook(get_activation("avgpool"))
    
    return model, activation

model, activation = load_model()

# Query image transform with grayscale conversion
query_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# File uploader
uploaded_file = st.file_uploader("Upload query image", 
                               type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Process query image
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=256)
        
        # Transform and extract features
        img_tensor = query_transform(img).unsqueeze(0)
        with torch.no_grad():
            _ = model(img_tensor)
            query_vec = activation["avgpool"].numpy().squeeze()[None, ...]
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        # Load database
        vecs = np.load("all_vecs.npy")
        names = np.load("all_names.npy")
        
        # Calculate similarities
        similarities = np.dot(vecs, query_vec.T).squeeze()
        top5 = np.argsort(similarities)[::-1][0:6]  # Exclude self-match
        
        # Display results
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.image(Image.open(f"./software_data/{names[top5[i]]}"), 
                       caption=f"Similarity: {similarities[top5[i]]:.2f}")
                
    except Exception as e:
        st.error(f"Error processing image: {e}")