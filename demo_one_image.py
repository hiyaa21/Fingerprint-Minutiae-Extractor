import cv2
import numpy as np
import math
import os # <-- Need os just for 'os.path.join'

# --- (Your Helper Functions: compute_crossing_number, get_distance) ---
def compute_crossing_number(neighborhood):
    """
    Compute the Crossing Number (CN) of a 3x3 neighborhood.
    """
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

# === 1. DEFINE THE IMAGE TO DEMO ===
# Change this to any image you want to test!
image_to_demo = os.path.join('real_data', '00000.bmp')
print(f"--- Processing {image_to_demo} for demo ---")

# === 2. RUN THE FULL PIPELINE (Steps 1-4) ===
img = cv2.imread(image_to_demo, 0)
if img is None:
    print(f"Error: Could not read {image_to_demo}.")
    exit()

enhanced_img = cv2.equalizeHist(img)
binary_img = cv2.adaptiveThreshold(
    enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 11, 2
)
kernel = np.ones((3, 3), np.uint8)
opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
cleaned_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel)
skeleton = cv2.ximgproc.thinning(cleaned_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

# --- (Step 4.5) Island Pruning ---
pruned_skeleton = np.zeros_like(skeleton)
min_ridge_area = 30
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > min_ridge_area:
        pruned_skeleton[labels == i] = 255

# --- (Step 5) Find ALL Minutiae ---
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

# --- (Step 5.5) Spur Pruning ---
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

# --- (Step 5.6) Cluster Pruning ---
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

# --- (Step 6) Draw Visualization Images ---

# Image 1: Raw (Noisy) Minutiae - This shows the "problem"
raw_minutiae_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
for (x, y) in raw_ridge_endings:
    cv2.circle(raw_minutiae_img, (x, y), 3, (0, 0, 255), -1) # Red
for (x, y) in raw_bifurcations:
    cv2.circle(raw_minutiae_img, (x, y), 3, (255, 0, 0), -1) # Blue

# Image 2: Final Pruned Minutiae - This shows the "solution"
final_minutiae_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
for (x, y) in pruned_ridge_endings:
    cv2.circle(final_minutiae_img, (x, y), 3, (0, 0, 255), -1) # Red
for (x, y) in final_bifurcations:
    cv2.circle(final_minutiae_img, (x, y), 3, (255, 0, 0), -1) # Blue

# --- (Step 7) Display All Results ---
print("Displaying results. Press any key to close all windows.")

# Make windows resizable
cv2.namedWindow("1. Enhanced Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("2. Original Skeleton (Noisy)", cv2.WINDOW_NORMAL)
cv2.namedWindow("3. Pruned Skeleton (Clean)", cv2.WINDOW_NORMAL)
cv2.namedWindow("4. Raw Minutiae (THE PROBLEM)", cv2.WINDOW_NORMAL)
cv2.namedWindow("5. Final Minutiae (THE SOLUTION)", cv2.WINDOW_NORMAL)

# Show all the images
cv2.imshow("1. Enhanced Image", enhanced_img)
cv2.imshow("2. Original Skeleton (Noisy)", skeleton)
cv2.imshow("3. Pruned Skeleton (Clean)", pruned_skeleton)
cv2.imshow("4. Raw Minutiae (THE PROBLEM)", raw_minutiae_img)
cv2.imshow("5. Final Minutiae (THE SOLUTION)", final_minutiae_img)

cv2.waitKey(0) # Wait for you to press a key
cv2.destroyAllWindows()