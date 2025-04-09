# Maize Disease Classification System

A real-time computer vision system for detecting and classifying maize (corn) diseases using deep learning. The system can identify four different conditions:
- Healthy Maize
- Gray Leaf Spot
- Common Rust
- Northern Leaf Blight

## Features

- Real-time disease classification using webcam
- Pre-trained CNN model for accurate disease detection
- Live confidence scores and FPS counter
- Support for both CPU and GPU inference
- Easy-to-use training pipeline for custom datasets

## Requirements

- Python 3.7+
- PyTorch 2.0+
- CUDA-capable GPU (optional, but recommended for better performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/3TH3M/maize_disease_classification.git
cd maize-disease-classification
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:
```
maize_dataset/
    ├── Healthy/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Gray_Leaf_Spot/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Common_Rust/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── Blight/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Usage

### Training the Model

1. Prepare your dataset as described above
2. Run the training script:
```bash
python train.py
```

The training process will:
- Split the dataset into training and testing sets
- Train the CNN model
- Save the trained model as `maize_classifier.pth`
- Generate training history plots

### Real-time Classification

To use the webcam for real-time classification:
```bash
python real_time_classification.py
```

Controls:
- Press 'q' to quit the application

The display shows:
- Current prediction and confidence score
- FPS (Frames Per Second) counter
- Color-coded confidence (Red to Green)

## Model Architecture

The system uses a CNN (Convolutional Neural Network) with the following architecture:
- 3 convolutional layers with ReLU activation
- Max pooling layers
- Dropout for regularization
- 2 fully connected layers
- Output layer with 4 classes

## Performance Tips

For best classification results:
1. Ensure good lighting conditions
2. Hold the leaf steady and centered in the frame
3. Maintain a consistent distance from the camera
4. Use a clean, neutral background
5. Use a GPU for faster inference

## Files Description

- `train.py`: Script for training the model
- `test.py`: Contains model architecture and testing functions
- `real_time_classification.py`: Real-time webcam classification
- `requirements.txt`: List of Python dependencies
- `maize_classifier.pth`: Trained model weights (generated after training)

## Troubleshooting

1. **Camera not working:**
   - Check if the correct camera index is selected (default is 0)
   - Ensure no other application is using the camera

2. **Low FPS:**
   - Consider using a GPU
   - Reduce camera resolution in `real_time_classification.py`
   - Close other resource-intensive applications

3. **Installation Issues:**
   - Ensure CUDA is properly installed for GPU support
   - Try creating a fresh virtual environment
   - Check Python version compatibility

I will update this as developments continue.
