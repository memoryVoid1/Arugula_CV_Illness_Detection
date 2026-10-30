import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import pandas as pd


# ==========================================
# --- TUNABLE PARAMETERS CONFIGURATION ---
# ==========================================

# File I/O
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 1. Noise Reduction (cv2.bilateralFilter)
BILATERAL_D = 20
BILATERAL_SIGMA_COLOR = 15
BILATERAL_SIGMA_SPACE = 120

# 2. Vegetation Segmentation (cv2.threshold - Otsu on Saturation)
OTSU_MIN_VAL = 0
MASK_MAX_VAL = 255  # Used for both Otsu and Purple thresholding

# 3. Lesion/Symptom Detection (cv2.threshold - LAB 'a' channel)
PURPLE_A_CHANNEL_MIN = 140

# 4. Math / Metrics Computations
BACKGROUND_PIXEL_VALUE = [0, 0, 0]  # Used to filter out mask background in np.any()
PERCENTAGE_MULTIPLIER = 100         # Used for printing the final infection percentage

# 1. Initialize an empty list before your image processing loop
results_data = []


def imageFactory (source,destination):
    source_dir = Path(source)
    destination_dir = Path(destination)

    if destination_dir.exists() and destination_dir.is_dir():
        print(f"Clearing all existing files in: {destination_dir.name}")
        shutil.rmtree(destination_dir)


    destination_dir.mkdir(parents=True, exist_ok=True)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for source_file in source_dir.iterdir():
        if source_file.is_file() and source_file.suffix.lower() in valid_extensions:
            print(f"Processing: {source_file.name}")
        
            image = cv2.imread(str(source_file))
        
            if image is None:
                print(f"Skipping (could not read): {source_file.name}")
                continue
            new_file_path = destination_dir / source_file.name
            altered = cv2.bilateralFilter(image,d=BILATERAL_D, sigmaColor=BILATERAL_SIGMA_COLOR, sigmaSpace=BILATERAL_SIGMA_SPACE)
            image_hsv= cv2.cvtColor(altered, cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(image_hsv)

            thresh_val, outputMask = cv2.threshold(s,OTSU_MIN_VAL,MASK_MAX_VAL,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            final = cv2.bitwise_and(image_hsv,image_hsv,mask=outputMask)
            
            lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)

            L, a, b = cv2.split(lab)
        
            _,purple_mask = cv2.threshold(a, PURPLE_A_CHANNEL_MIN, MASK_MAX_VAL, cv2.THRESH_BINARY)                       
        
            valid_symptoms = cv2.bitwise_and(purple_mask,outputMask)
            symptoms_visualized = cv2.bitwise_and(image, image, mask=valid_symptoms)
            cv2.imwrite(str(new_file_path), symptoms_visualized)
            
            non_black_pixels = np.any(symptoms_visualized != BACKGROUND_PIXEL_VALUE, axis=-1)
            non_black_count = np.sum(non_black_pixels)
            plant_pixels = np.any(final != BACKGROUND_PIXEL_VALUE, axis=-1)
            all_plant_count = np.sum(plant_pixels)

            infectionPercentage = (non_black_count/all_plant_count)*100
            print(infectionPercentage)

            
            image_name = str(source_file.name)
            severity_ratio = infectionPercentage
            severity_level = "Medium"

            results_data.append({
            "image_name": image_name,
            "severity_ratio": severity_ratio,
            "severity_level": severity_level
            })

    df = pd.DataFrame(results_data)

    df.to_excel("results.xlsx", index=False, engine='openpyxl')
    
    pipeline_content = f"""Execution Order and Parameters:
    1. Noise Reduction: cv2.bilateralFilter (d={BILATERAL_D}, sigmaColor={BILATERAL_SIGMA_COLOR}, sigmaSpace={BILATERAL_SIGMA_SPACE})
    2. Color Conversion (Vegetation): BGR to HSV
    3. Vegetation Segmentation: cv2.threshold with Otsu on HSV Saturation channel (thresh_min={OTSU_MIN_VAL}, maxval={MASK_MAX_VAL})
    4. Color Conversion (Symptoms): BGR to LAB
    5. Lesion Detection: cv2.threshold on LAB 'a' channel for purpleness (thresh_min={PURPLE_A_CHANNEL_MIN}, maxval={MASK_MAX_VAL})
    6. Mask Refinement: Logical AND combining purple lesion mask with vegetation mask
    7. Severity Computation: Ratio of symptom pixels to total plant area pixels (multiplier={PERCENTAGE_MULTIPLIER})
    """
    
    with open("pipeline.txt", "w") as f:
        f.write(pipeline_content)
    return 1


imageFactory("Images/Arugula","Results")

# Ödevde geçen Stage E — Cluster-based level calibration and assignment kısmı kaldı, anlam veremediğin kısım varsa sor.