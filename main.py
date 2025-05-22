import os
from PIL import Image
import numpy as np

import torch
import torchvision
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = 933120000
Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb protection

data_dir = "./software_data"
images = os.listdir(data_dir)

model = torchvision.models.resnet18(weights="DEFAULT")
model.eval()

all_names = []
all_vecs = None

# Update transform to use 224x224 and normalize features
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Changed to standard ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_activation("avgpool"))

with torch.no_grad():
    for i, file in enumerate(images):
        try:
            img_path = os.path.join(data_dir, file)
            img = Image.open(img_path).convert("RGB")  # <- convert to RGB to avoid issues
            img = transform(img)
            # Extract and normalize vector
            _ = model(img.unsqueeze(0))
            vec = activation["avgpool"].numpy().squeeze()[None, ...]
            vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)  # L2 normalization  
            all_vecs = vec if all_vecs is None else np.vstack([all_vecs, vec])
            all_names.append(file)
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue
        if i % 100 == 0 and i != 0:
            print(i, "done")

np.save("all_vecs.npy", all_vecs)
np.save("all_names.npy", all_names)
