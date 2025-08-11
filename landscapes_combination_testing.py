#Haben Haile, Chloe Ho, William Andreopoulos
#SJSU Research Internship Project
#Differentiation of Human and AI-Generated Images Across Several Image Categories
#August 15, 2025

import os
import numpy as np
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import csv


# List of features that will be tested in combinations
features = [
    'color_histogram',
    'lbp',
    'haralick',
    'edge_density',
    'dft',
    'wavelet',
    'multiscale_wavelet',
    'sharpness',
    'entropy',
    'color_moments',
    'blur',
    'gradient_stats',
    'jpeg_artifacts',
    'tamura_texture',
    'color_palette_diversity',
    'hue_saturation_stats',
    'layout_alignment',
    'color_region_count'
]

# Load in the feature data

feature_data = {}
labels = []

print("Loading all feature vectors from .npy files...")
for feat in features:
    path = f'features/{feat}.npy'
    feature_data[feat] = np.load(path)

# Create true labels only once (500 AI-generated, 100 human)
if not labels:
    labels = np.array([0] * 500 + [1] * 100)

results = []

# Test all possible combinations of the features 
print("Testing every possible combination of the features\n")

# Try every possible combination of 1 to all features
for r in range(1, len(features) + 1):
    for combo in itertools.combinations(features, r):
        # Merge features horizontally for this combination
        combined = np.concatenate([feature_data[feat] for feat in combo], axis=1)

        # Normalize the features
        scaler = StandardScaler()
        combined = scaler.fit_transform(combined)

        # Reduce dimensionality using PCA (retain 95% variance)
        pca = PCA(n_components=0.95)
        combined = pca.fit_transform(combined)

        # Apply KMeans clustering (unsupervised)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        preds = kmeans.fit_predict(combined)

        
        # Manually match cluster labels to 0/1 by majority voting
        cluster_0 = [0, 0]
        cluster_1 = [0, 0] 
        for i, label in enumerate(labels):
            if preds[i] == 0:
                cluster_0[label] += 1
            else:
                cluster_1[label] += 1

        # Decide which cluster corresponds to which label
        if cluster_0[0] + cluster_1[1] > cluster_1[0] + cluster_0[1]:
            cluster_to_class = {0: 0, 1: 1}
        else:
            cluster_to_class = {0: 1, 1: 0}

        # Map predictions to label classes
        remapped_preds = np.array([cluster_to_class[p] for p in preds])

        # Evaluation metrics
        acc = np.mean(remapped_preds == labels)
        f1 = f1_score(labels, remapped_preds)
        ari = adjusted_rand_score(labels, remapped_preds)

        # Breakdown of correct/incorrect predictions by class
        correct_ai = np.sum((labels == 0) & (remapped_preds == 0))
        correct_real = np.sum((labels == 1) & (remapped_preds == 1))
        incorrect_ai = 500 - correct_ai
        incorrect_real = 100 - correct_real
        recall_ai = correct_ai / 500
        recall_real = correct_real / 100

        # Balanced accuracy - average recall of both classes
        balanced_accuracy = 0.5 * (recall_ai + recall_real)

        # Store the results for this combination
        results.append({
            'features': ', '.join(combo),
            'accuracy': acc,
            'f1_score': f1,
            'adjusted_rand_index': ari,
            'balanced_accuracy': balanced_accuracy,
            'correct_ai': correct_ai,
            'correct_real': correct_real,
            'incorrect_ai': incorrect_ai,
            'incorrect_real': incorrect_real
        })

# Save top results

# Sort by balanced accuracy (most fair performance across both classes)
results.sort(key=lambda x: x['balanced_accuracy'], reverse=True)

# Save top 100 combinations (optional)
top_results = results[:100]

# Save all results to CSV file
csv_path = 'feature_combination_results.csv'
with open(csv_path, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=[
        'features',
        'accuracy',
        'f1_score',
        'adjusted_rand_index',
        'balanced_accuracy',
        'correct_ai',
        'correct_real',
        'incorrect_ai',
        'incorrect_real'
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"Done! All results saved to → {csv_path}")
print(f"Top combination: {top_results[0]['features']} — Balanced Accuracy: {top_results[0]['balanced_accuracy']:.4f}")
