import os
from PIL import Image
from torchvision import transforms
import random

# Define augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

# Augment images in a directory
def augment_images_in_directory(directory, num_augmentations=5):
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            
            for i in range(num_augmentations):
                augmented_image = augmentation_transforms(image)
                new_filename = f"aug_{i}_{filename}"
                augmented_image.save(os.path.join(directory, new_filename))

# Main function to augment the entire dataset
def main():
    base_dir = 'maize_dataset'
    classes = ['Healthy', 'Gray_Leaf_Spot', 'Common_Rust', 'Blight']
    
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        print(f"Augmenting images in {class_dir}...")
        augment_images_in_directory(class_dir)

if __name__ == "__main__":
    main() 