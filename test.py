import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the same model architecture as in training
class MaizeCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(MaizeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def predict_image(image_path, model, transform, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model = MaizeCNN().to(device)
    model.load_state_dict(torch.load('maize_classifier.pth'))
    
    # Define the same transformations as in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Class names
    classes = ['Healthy', 'Gray_Leaf_Spot', 'Common_Rust', 'Blight']
    
    # Test on images from each class
    print("\nTesting model on sample images from each class:")
    print("-" * 50)
    
    for class_name in classes:
        class_dir = os.path.join("maize_dataset", class_name)
        if not os.path.exists(class_dir):
            continue
            
        # Get the first image from each class
        image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue
            
        test_image = os.path.join(class_dir, image_files[0])
        predicted_class, confidence = predict_image(test_image, model, transform, device)
        
        print(f"Testing image from: {class_name}")
        print(f"Predicted class: {classes[predicted_class]}")
        print(f"Confidence: {confidence:.2%}")
        print("-" * 50)

if __name__ == "__main__":
    main() 