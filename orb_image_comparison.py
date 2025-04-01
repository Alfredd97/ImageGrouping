import cv2
import numpy as np
import os
from collections import defaultdict

def load_and_preprocess_image(image_path):
    # Read image directly in grayscale
    print(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(img)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(equalized)
    
    return enhanced

def compare_images_orb(img1_path, img2_path, threshold=0.3, show_matches=False):
    # Load and preprocess images
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    
    if img1 is None or img2 is None:
        return False
    
    # Create ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return False
    
    # Create BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate similarity score
    num_matches = len(matches)
    min_features = max(len(kp1), len(kp2))
    similarity_score = num_matches / min_features
    
    # Visualize matches if requested
    if show_matches:
        # Convert grayscale images to BGR for visualization
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Draw matches
        result_img = cv2.drawMatches(img1_color, kp1, img2_color, kp2, 
                                   matches[:50], None, 
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Show the result
        cv2.imshow("ORB Feature Matching", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(similarity_score)
    return similarity_score > threshold

def find_duplicate_images(folder_path, show_matches=False):
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
            
            if compare_images_orb(img1_path, img2_path, show_matches=show_matches):
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
    
    # Ask user if they want to see the matches
    show_matches = input("Do you want to see the feature matches? (y/n): ").lower() == 'y'
    
    # Find duplicate images
    similar_images = find_duplicate_images(folder_path, show_matches)
    
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