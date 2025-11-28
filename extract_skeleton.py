import cv2
import numpy as np
import glob
import os
import math
import csv  # <-- 1. IMPORT THE CSV LIBRARY

# ... (keep your compute_crossing_number and get_distance functions here) ...
def compute_crossing_number(neighborhood):
    p1 = neighborhood[0, 0]; p2 = neighborhood[0, 1]; p3 = neighborhood[0, 2]
    p8 = neighborhood[1, 0];                        p4 = neighborhood[1, 2]
    p7 = neighborhood[2, 0]; p6 = neighborhood[2, 1]; p5 = neighborhood[2, 2]
    neighbors = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p1]) / 255
    transitions = sum(abs(neighbors[i+1] - neighbors[i]) for i in range(8))
    cn = transitions / 2
    return int(cn)

def get_distance(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# === 1. SETUP FOLDERS ===
data_folders = ['real_data', 'train_data'] 
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# === 2. PREPARE FOR RESULTS MATRIX ===
# This list will hold all our data for the CSV
results_data = [] 
# This is the Header Row for our CSV file (our "Matrix")
csv_header = [
    "Image Name", 
    "Raw Endings (Noisy)", "Raw Bifurcations (Noisy)", 
    "Spur-Pruned Endings", "Spur-Pruned Bifurcations",
    "Final Endings (Clean)", "Final Bifurcations (Clean)"
]

print(f"Starting batch processing... Output will be saved to '{output_dir}'")
print(f"Comparison matrix will be saved to 'comparison_matrix.csv'")

# === 3. LOOP THROUGH ALL IMAGES ===
for folder in data_folders:
    image_paths = glob.glob(os.path.join(folder, "*.bmp"))
    print(f"--- Processing {len(image_paths)} images in '{folder}' ---")
    
    for img_path in image_paths:
        
        # ... (Steps 1-4.5: Load, Enhance, Binarize, Clean, Thin, Island Prune) ...
        img = cv2.imread(img_path, 0)
        if img is None:
            print(f"Error: Could not read {img_path}. Skipping.")
            continue
        enhanced_img = cv2.equalizeHist(img)
        binary_img = cv2.adaptiveThreshold(
            enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        cleaned_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)
        skeleton = cv2.ximgproc.thinning(cleaned_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        pruned_skeleton = np.zeros_like(skeleton)
        min_ridge_area = 30
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_ridge_area:
                pruned_skeleton[labels == i] = 255

        # ... (Step 5: Find Raw Minutiae) ...
        raw_ridge_endings = []
        raw_bifurcations = []
        h, w = pruned_skeleton.shape
        margin = 10 
        for y in range(margin, h - margin):
            for x in range(margin, w - margin):
                if pruned_skeleton[y, x] == 255:
                    neighborhood = pruned_skeleton[y-1:y+2, x-1:x+2]
                    cn = compute_crossing_number(neighborhood)
                    if cn == 1:
                        raw_ridge_endings.append((x, y))
                    elif cn == 3:
                        raw_bifurcations.append((x, y))
        
        # ... (Step 5.5: Spur Pruning) ...
        spur_threshold = 15 
        pruned_ridge_endings = []
        pruned_bifurcations = []
        endings_to_remove = set()
        bifurcations_to_remove = set()
        for r_idx, r_point in enumerate(raw_ridge_endings):
            for b_idx, b_point in enumerate(raw_bifurcations):
                if b_idx in bifurcations_to_remove: continue
                dist = get_distance(r_point, b_point)
                if dist < spur_threshold:
                    endings_to_remove.add(r_idx)
                    bifurcations_to_remove.add(b_idx)
                    break 
        for r_idx, r_point in enumerate(raw_ridge_endings):
            if r_idx not in endings_to_remove:
                pruned_ridge_endings.append(r_point)
        for b_idx, b_point in enumerate(raw_bifurcations):
            if b_idx not in bifurcations_to_remove:
                pruned_bifurcations.append(b_point)

        # ... (Step 5.6: Cluster Pruning) ...
        cluster_threshold = 10
        final_bifurcations = []
        bifurcations_to_remove_cluster = set()
        for i in range(len(pruned_bifurcations)):
            if i in bifurcations_to_remove_cluster: continue
            b_point_1 = pruned_bifurcations[i]
            for j in range(i + 1, len(pruned_bifurcations)):
                if j in bifurcations_to_remove_cluster: continue
                b_point_2 = pruned_bifurcations[j]
                dist = get_distance(b_point_1, b_point_2)
                if dist < cluster_threshold:
                    bifurcations_to_remove_cluster.add(i)
                    bifurcations_to_remove_cluster.add(j)
        for i in range(len(pruned_bifurcations)):
            if i not in bifurcations_to_remove_cluster:
                final_bifurcations.append(pruned_bifurcations[i])

        # === 4. SAVE DATA FOR MATRIX ===
        # Get the simple name (e.g., "00000.bmp")
        base_name = os.path.basename(img_path) 
        
        # Get the counts from each stage
        num_raw_endings = len(raw_ridge_endings)
        num_raw_bifurcations = len(raw_bifurcations)
        num_spur_pruned_endings = len(pruned_ridge_endings)
        num_spur_pruned_bifurcations = len(pruned_bifurcations)
        num_final_endings = len(pruned_ridge_endings) # Stays the same
        num_final_bifurcations = len(final_bifurcations)
        
        # Add all data for this image to our results list
        results_data.append([
            base_name,
            num_raw_endings, num_raw_bifurcations,
            num_spur_pruned_endings, num_spur_pruned_bifurcations,
            num_final_endings, num_final_bifurcations
        ])

        # ... (Step 6 & 7: Draw and Save Image) ...
        minutiae_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
        for (x, y) in pruned_ridge_endings:
            cv2.circle(minutiae_img, (x, y), 3, (0, 0, 255), -1)
        for (x, y) in final_bifurcations:
            cv2.circle(minutiae_img, (x, y), 3, (255, 0, 0), -1)
        
        name_only = os.path.splitext(base_name)[0]
        output_filename = os.path.join(output_dir, f"{name_only}_minutiae.png")
        cv2.imwrite(output_filename, minutiae_img)

print(f"--- Batch processing complete. ---")

# === 5. WRITE THE FINAL MATRIX TO A CSV FILE ===
csv_filename = 'comparison_matrix.csv'
try:
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header) # Write the header row
        writer.writerows(results_data) # Write all the data rows
    print(f"Successfully created comparison matrix: {csv_filename}")
except Exception as e:
    print(f"Error: Could not write CSV file. {e}")