import cv2
import numpy as np
import os
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim

def load_and_preprocess_image(image_path):
    # Read image
    print("Reading Image", image_path)
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Resize image to a standard size for comparison
    resized = cv2.resize(equalized, (224, 224))
    
    # Apply additional contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(resized)
    
    return enhanced

def compare_images_ssim(img1_path, img2_path, threshold=0.2):
    # Load and preprocess images
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    
    if img1 is None or img2 is None:
        return False
    
    # Calculate SSIM
    similarity = ssim(img1, img2)
    print(similarity)
    return similarity > threshold

def find_duplicate_images(folder_path):
    # Dictionary to store similar images
    similar_images = defaultdict(list)
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Keep track of which images have been grouped
    grouped_images = set()
    
    # Compare each pair of images
    for i in range(len(image_files)):
        img1_path = os.path.join(folder_path, image_files[i])
        
        # Check if this image is already in a group
        already_grouped = False
        for group in similar_images.values():
            if image_files[i] in group:
                already_grouped = True
                break
        
        if already_grouped:
            continue
        
        # Create a new group for this image
        current_group = [image_files[i]]
        grouped_images.add(image_files[i])
        
        for j in range(i + 1, len(image_files)):
            img2_path = os.path.join(folder_path, image_files[j])
            
            if compare_images_ssim(img1_path, img2_path):
                current_group.append(image_files[j])
                grouped_images.add(image_files[j])
        
        # Always add the group, even if it has only one image
        similar_images[f"group_{len(similar_images) + 1}"] = current_group
    
    # Add any remaining ungrouped images to their own groups
    for image in image_files:
        if image not in grouped_images:
            similar_images[f"group_{len(similar_images) + 1}"] = [image]
    
    return similar_images

def main():
    # Path to the images folder
    folder_path = "images"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist!")
        return
    
    # Find duplicate images
    similar_images = find_duplicate_images(folder_path)
    
    # Print results
    if similar_images:
        print("\nFound similar image groups:")
        for group_name, images in similar_images.items():
            print(f"\n{group_name}:")
            for img in images:
                print(f"  - {img}")
    else:
        print("\nNo similar images found!")

if __name__ == "__main__":
    main() 