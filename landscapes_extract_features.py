#Haben Haile, Chloe Ho, William Andreopoulos
#SJSU Research Internship Project
#Differentiation of Human and AI-Generated Images Across Several Image Categories
#August 15, 2025

#Dataset created by Haben Haile: https://drive.google.com/drive/folders/1Eeitymcr_84lKR9eYi2WYLEU92pF22eg?usp=sharing

import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy
from scipy.stats import skew
import mahotas
import pywt
from sklearn.cluster import KMeans

# This sets up the folder paths
IMG_ROOT = ''
SAVE_DIR = 'features'
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize the images and labels variables
images = []
labels = []

print("Scanning dataset folders...")
for i in range(1, 101):
    folder = os.path.join(IMG_ROOT, str(i))

    # Load real images, label = 1
    real_path = os.path.join(folder, 'Real')
    for fname in os.listdir(real_path):
        img = cv2.imread(os.path.join(real_path, fname))
        if img is not None:
            images.append(img)
            labels.append(1)

    # Load AI-generated images, label = 0
    ai_path = os.path.join(folder, 'AI')
    for fname in sorted(os.listdir(ai_path)):
        img = cv2.imread(os.path.join(ai_path, fname))
        if img is not None:
            images.append(img)
            labels.append(0)

print(f"Loaded {len(images)} total images")

# === Feature Extractor Functions ===
# Each function extracts a different feature from the human made images

def extract_color_histogram(img):
    hist = []
    for i in range(3):  # BGR channels
        h = cv2.calcHist([img], [i], None, [64], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist.extend(h)
    return hist

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_haralick(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray, return_mean=True).tolist()

def extract_edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return [np.sum(edges > 0) / edges.size]

def extract_dft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(gray)
    mag = np.abs(dft)
    return [np.mean(mag), np.var(mag)]

def extract_wavelet(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs
    return [np.mean(cA), np.mean(cH), np.mean(cV), np.mean(cD)]

def extract_multiscale_wavelet(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stats = []
    coeffs = pywt.wavedec2(gray, 'haar', level=3)
    for cH, cV, cD in coeffs[1:]:
        stats.extend([
            np.mean(cH), np.std(cH),
            np.mean(cV), np.std(cV),
            np.mean(cD), np.std(cD)
        ])
    return stats

def extract_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return [cv2.Laplacian(gray, cv2.CV_64F).var()]

def extract_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return [shannon_entropy(gray)]

def extract_color_moments(img):
    moments = []
    for i in range(3):
        channel = img[:, :, i].flatten()
        moments.extend([np.mean(channel), np.std(channel), skew(channel)])
    return moments

def extract_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return [cv2.Laplacian(gray, cv2.CV_64F).var()]

def extract_gradient_stats(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    return [np.mean(mag), np.var(mag)]

def extract_jpeg_artifacts(img):
    temp_path = "temp_compress.jpg"
    cv2.imwrite(temp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    jpeg_img = cv2.imread(temp_path)
    os.remove(temp_path)
    diff = cv2.absdiff(img, jpeg_img)
    return [np.mean(diff), np.std(diff)]

def extract_tamura_texture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = mahotas.features.haralick(gray)
    contrast = glcm[:, 1].mean()
    homogeneity = glcm[:, 4].mean()
    return [contrast, homogeneity]

def extract_color_palette_diversity(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reshaped = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(reshaped)
    return list(np.sort(np.bincount(kmeans.labels_) / len(kmeans.labels_)))

def extract_hue_saturation_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].flatten()
    sat = hsv[:, :, 1].flatten()
    return [np.mean(hue), np.std(hue), np.mean(sat), np.std(sat)]

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

# This a list of all the features extracted with corresponding names for the functions
feature_funcs = {
    'color_histogram': extract_color_histogram,
    'lbp': extract_lbp,
    'haralick': extract_haralick,
    'edge_density': extract_edge_density,
    'dft': extract_dft,
    'wavelet': extract_wavelet,
    'multiscale_wavelet': extract_multiscale_wavelet,
    'sharpness': extract_sharpness,
    'entropy': extract_entropy,
    'color_moments': extract_color_moments,
    'blur': extract_blur,
    'gradient_stats': extract_gradient_stats,
    'jpeg_artifacts': extract_jpeg_artifacts,
    'tamura_texture': extract_tamura_texture,
    'color_palette_diversity': extract_color_palette_diversity,
    'hue_saturation_stats': extract_hue_saturation_stats,
    'layout_alignment': extract_layout_alignment,
    'color_region_count': extract_color_region_count
}

# === Run Feature Extraction ===
print("Starting feature extraction...")
# This for loop goes through every extraction function and runs the function
for name, func in feature_funcs.items():
    features = []
    for img in tqdm(images, desc=f"Extracting: {name}"):
        try:
            feat = func(img)
            features.append(feat)
        except Exception as e:
            print(f"Failed to extract {name} features: {e}")
            features.append([0] * len(func(images[0])))  # fallback on error
    np.save(os.path.join(SAVE_DIR, f'{name}.npy'), np.array(features))

# Save labels
np.save(os.path.join(SAVE_DIR, 'labels.npy'), np.array(labels))
print("Features and labels are saved to the 'features/' folder.")
