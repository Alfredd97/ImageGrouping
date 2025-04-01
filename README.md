# Image Grouping

A Python tool for detecting and grouping similar images using two different methods:
1. ORB (Oriented FAST and Rotated BRIEF) feature matching
2. SSIM (Structural Similarity Index Measure)

## Features

- Detects similar images regardless of brightness differences
- Uses histogram equalization and CLAHE for better contrast handling
- Supports multiple image formats (PNG, JPG, JPEG, BMP, GIF)
- Visualizes feature matches (for ORB method)
- Groups similar images together
- Handles single images in their own groups

## Requirements

- Python 3.9+
- OpenCV
- NumPy
- scikit-image

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Alfredd97/ImageGrouping.git
cd ImageGrouping
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your images in the `images` folder

2. Run either script:
```bash
# Using ORB method
python orb_image_comparison.py

# Using SSIM method
python ssim_image_comparison.py
```

3. For the ORB method, you'll be prompted to show feature matches (y/n)

## Output

The scripts will group similar images together and display the results in the following format:
```
Found similar image groups:

group_1:
  - image1.jpg
  - image2.jpg
  - image3.jpg

group_2:
  - image4.jpg
  - image5.jpg

group_3:
  - image6.jpg
```

## Notes

- ORB method threshold: 0.3 (30% similarity)
- SSIM method threshold: 0.2 (20% similarity)
- You can adjust these thresholds in the respective scripts 