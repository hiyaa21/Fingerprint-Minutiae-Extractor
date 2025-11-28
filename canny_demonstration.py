import cv2
import numpy as np
import os

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

# Define the path to a sample image
image_path = os.path.join('real_data', '00000.bmp')
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

print(f"Processing {image_path} with Canny Edge Detection...")

# 1. Load the image
img = cv2.imread(image_path, 0)
if img is None:
    print(f"Error: Could not read {image_path}. Make sure the file exists.")
    exit()

# 2. Enhance (Histogram Equalization)
enhanced_img = cv2.equalizeHist(img)

# 3. Apply Canny Edge Detection
# Canny gives us White Edges on Black Background (which looks like a skeleton)
# BUT these are boundaries, not centerlines.
edges = cv2.Canny(enhanced_img, 100, 200)

# 4. Attempt Minutiae Detection on Canny Edges
# We treat the Canny edges as if they were a "skeleton" to show why it fails.
minutiae_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
ridge_endings = []
bifurcations = []

h, w = edges.shape
margin = 10

for y in range(margin, h - margin):
    for x in range(margin, w - margin):
        # Check if this pixel is an edge
        if edges[y, x] == 255:
            neighborhood = edges[y-1:y+2, x-1:x+2]
            cn = compute_crossing_number(neighborhood)
            
            if cn == 1:
                ridge_endings.append((x, y))
            elif cn == 3:
                bifurcations.append((x, y))

# 5. Draw the False Minutiae
print(f"Detected {len(ridge_endings)} Endings and {len(bifurcations)} Bifurcations on Canny Edges.")

for (x, y) in ridge_endings:
    cv2.circle(minutiae_img, (x, y), 3, (0, 0, 255), -1) # Red

for (x, y) in bifurcations:
    cv2.circle(minutiae_img, (x, y), 3, (255, 0, 0), -1) # Blue

# 6. Save the result
output_filename = os.path.join(output_dir, 'canny_failure_demonstration.png')
cv2.imwrite(output_filename, minutiae_img)

print(f"Saved Canny failure demonstration to {output_filename}")