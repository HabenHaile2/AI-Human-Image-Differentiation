import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from sklearn.cluster import KMeans
from scipy.stats import entropy as scipy_entropy


# Path for dataset which contains real and AI infographic images
IMG_ROOT = ''

# Folder where extracted features and labels are saved
SAVE_DIR = 'features_infographics'
# Create the save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# LOAD IMAGES AND ASSIGN LABELS

images = []  # List to hold image data
labels = []  # Corresponding labels: 0 = AI-generated, 1 = human-created

# Loop through folder 1 to 100 and read AI and human images
for i in range(1, 101):
    folder = os.path.join(IMG_ROOT, str(i))

    # Load AI-generated image
    ai_path = os.path.join(folder, 'gemini-image.png')
    ai_img = cv2.imread(ai_path)
    if ai_img is not None:
        images.append(ai_img)
        labels.append(0)  # Label for AI

    # Load human-created image
    real_path = os.path.join(folder, 'human-image.png')
    real_img = cv2.imread(real_path)
    if real_img is not None:
        images.append(real_img)
        labels.append(1)  # Label for human

print(f"Loaded {len(images)} images successfully.")

# === FEATURE EXTRACTOR FUNCTIONS ===

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) 
    return hist

def extract_gradient_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    return [np.mean(mag), np.var(mag)]

def extract_layout_alignment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return [0, 0] 
    centers = np.array([cv2.boundingRect(c) for c in contours])
    xs = centers[:, 0] 
    ys = centers[:, 1]  
    return [np.var(xs), np.var(ys)]  

def extract_color_region_count(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reshaped = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reshaped)
    counts = np.bincount(labels)
    return (counts / counts.sum()).tolist()

def extract_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / hist.sum()
    return [scipy_entropy(hist, base=2)]

def extract_edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return [np.mean(edges > 0)] 

def extract_hue_saturation_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    return [np.mean(h), np.var(h), np.mean(s), np.var(s)]

def extract_color_palette_diversity(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reshaped = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(reshaped)
    return list(np.sort(np.bincount(kmeans.labels_) / len(kmeans.labels_))) 

def extract_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return [lap.var()]

# This a list of all the features extracted with corresponding names for the functions
feature_funcs = {
    'lbp': extract_lbp,
    'gradient_stats': extract_gradient_stats,
    'layout_alignment': extract_layout_alignment,
    'color_region_count': extract_color_region_count,
    'entropy': extract_entropy,
    'edge_density': extract_edge_density,
    'hue_saturation_stats': extract_hue_saturation_stats,
    'color_palette_diversity': extract_color_palette_diversity,
    'blur': extract_blur
}

# === FEATURE EXTRACTION LOOP ===

print("\n Starting feature extraction...\n")

# Loop through all feature functions
for name, func in feature_funcs.items():
    features = []  # Store feature vectors for all images
    for img in tqdm(images, desc=f"Extracting '{name}' features"):
        try:
            feat = func(img)
            features.append(feat)
        except Exception as e:
            # Log and handle unexpected errors without crashing
            print(f"Skipping one image in '{name}' due to error: {e}")
            features.append([0] * len(func(images[0])))
    features = np.array(features)
    np.save(os.path.join(SAVE_DIR, f'{name}.npy'), features)  # Save as .npy file

# Save labels array
np.save(os.path.join(SAVE_DIR, 'labels.npy'), np.array(labels))

print("\n All features and labels have been saved in the 'features_infographics' directory.")
print("Done!")
