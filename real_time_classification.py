import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import time
from test import MaizeCNN  # Reusing our model architecture

def get_prediction(frame, model, transform, device):
    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Transform the image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item()

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = MaizeCNN().to(device)
    model.load_state_dict(torch.load('maize_classifier.pth'))
    model.eval()

    # Define the same transformations as in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Class names
    classes = ['Healthy', 'Gray_Leaf_Spot', 'Common_Rust', 'Blight']

    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' to quit")
    
    # FPS calculation variables
    fps_start_time = time.time()
    fps = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Calculate FPS
        frame_count += 1
        if frame_count >= 30:
            fps = frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            frame_count = 0

        # Get prediction
        predicted_class, confidence = get_prediction(frame, model, transform, device)
        
        # Draw prediction on frame
        label = f"{classes[predicted_class]}: {confidence:.2%}"
        
        # Set color based on confidence (Red to Green)
        color = (int(255 * (1 - confidence)), int(255 * confidence), 0)
        
        # Add text to frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Maize Disease Classification', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 