#Haben Haile, Chloe Ho, William Andreopoulos
#SJSU Research Internship Project
#Differentiation of Human and AI-Generated Images Across Several Image Categories
#August 15, 2025

import os
import csv
import numpy as np
import itertools
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    adjusted_rand_score,
)

# Directory containing features extracted from landscape images
LANDSCAPE_FEAT_DIR = 'features'

# Directory containing features extracted from infographic images
INFOGRAPHIC_FEAT_DIR = 'features_infographics'

# Path to save the summary CSV output
SUMMARY_PATH = 'cluster_summary.csv'

# List of features to evaluate in combinations
SELECTED_FEATURES = [
    'gradient_stats',
    'layout_alignment',
    'color_region_count',
    'lbp',
    'entropy',
    'edge_density',
    'hue_saturation_stats',
    'color_palette_diversity',
    'blur',
]

# This function loads feature vectors for a given directory and list of feature names
def load_feature_stack(dir_path, feature_names):
    feature_arrays = []
    for name in feature_names:
        file_path = os.path.join(dir_path, f"{name}.npy")
        arr = np.load(file_path)
        feature_arrays.append(arr)
    return np.concatenate(feature_arrays, axis=1)

# This function loads class labels and prefixes
def load_labels(label_path, label_prefix):
    labels = np.load(label_path)
    # Prefixing each label with 'Landscape_' or 'Infographic_' + class ('AI' or 'Human')
    return [f"{label_prefix}_Human" if l == 1 else f"{label_prefix}_AI" for l in labels]

# This function evaluates a single feature combination
def evaluate_features(feature_combo):
    # Load and combine features for both datasets
    X_landscape = load_feature_stack(LANDSCAPE_FEAT_DIR, feature_combo)
    y_landscape = load_labels(os.path.join(LANDSCAPE_FEAT_DIR, 'labels.npy'), 'Landscape')

    X_infographic = load_feature_stack(INFOGRAPHIC_FEAT_DIR, feature_combo)
    y_infographic = load_labels(os.path.join(INFOGRAPHIC_FEAT_DIR, 'labels.npy'), 'Infographic')

    # Combine the two datasets
    X = np.vstack([X_landscape, X_infographic])
    y_true = y_landscape + y_infographic

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X)

    # Map cluster IDs to the most common true label in each cluster
    cluster_to_true_labels = defaultdict(list)
    for cluster_id, true_label in zip(cluster_ids, y_true):
        cluster_to_true_labels[cluster_id].append(true_label)

    cluster_to_label = {
        cluster_id: Counter(labels).most_common(1)[0][0]
        for cluster_id, labels in cluster_to_true_labels.items()
    }

    # Reconstruct predicted labels using cluster-to-label map
    y_pred = [cluster_to_label[cid] for cid in cluster_ids]

    # Evaluate performance
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    ari = adjusted_rand_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Detailed breakdown by category
    correct_landscape_ai = correct_landscape_human = 0
    correct_infographic_ai = correct_infographic_human = 0
    incorrect_landscape_ai = incorrect_landscape_human = 0
    incorrect_infographic_ai = incorrect_infographic_human = 0

    for actual, predicted in zip(y_true, y_pred):
        is_correct = (actual == predicted)

        if "Landscape" in actual:
            if "AI" in actual:
                correct_landscape_ai += is_correct
                incorrect_landscape_ai += not is_correct
            else:
                correct_landscape_human += is_correct
                incorrect_landscape_human += not is_correct
        elif "Infographic" in actual:
            if "AI" in actual:
                correct_infographic_ai += is_correct
                incorrect_infographic_ai += not is_correct
            else:
                correct_infographic_human += is_correct
                incorrect_infographic_human += not is_correct

    return {
        'features': ', '.join(feature_combo),
        'accuracy': accuracy,
        'f1_score': f1,
        'adjusted_rand_index': ari,
        'balanced_accuracy': balanced_acc,
        'correct_landscape_ai': correct_landscape_ai,
        'correct_landscape_real': correct_landscape_human,
        'correct_infographic_ai': correct_infographic_ai,
        'correct_infographic_real': correct_infographic_human,
        'incorrect_landscape_ai': incorrect_landscape_ai,
        'incorrect_landscape_real': incorrect_landscape_human,
        'incorrect_infographic_ai': incorrect_infographic_ai,
        'incorrect_infographic_real': incorrect_infographic_human,
    }

# Main execution
def main():
    results = []

    print("Starting cluster evaluation for all feature combinations\n")

    # Try all non-empty combinations of selected features
    for r in range(1, len(SELECTED_FEATURES) + 1):
        for feature_combo in itertools.combinations(SELECTED_FEATURES, r):
            print(f"Testing feature combo: {feature_combo}")
            result = evaluate_features(feature_combo)
            results.append(result)

    # Sort results by highest balanced accuracy
    results.sort(key=lambda x: x['balanced_accuracy'], reverse=True)

    # Save results to CSV
    header = [
        'features',
        'accuracy',
        'f1_score',
        'adjusted_rand_index',
        'balanced_accuracy',
        'correct_landscape_ai',
        'correct_landscape_real',
        'correct_infographic_ai',
        'correct_infographic_real',
        'incorrect_landscape_ai',
        'incorrect_landscape_real',
        'incorrect_infographic_ai',
        'incorrect_infographic_real',
    ]

    with open(SUMMARY_PATH, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    print("\n Evaluation complete.")
    print(f"Results saved to: {SUMMARY_PATH}")
    print(f"Top performing combo: {results[0]['features']} (Balanced Accuracy = {results[0]['balanced_accuracy']:.4f})")

# === RUN SCRIPT ===
if __name__ == "__main__":
    main()