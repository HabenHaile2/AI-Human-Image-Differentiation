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
    adjusted_rand_score
)

# Directories with features
LANDSCAPE_FEAT_DIR = 'features'
INFOGRAPHIC_FEAT_DIR = 'features_infographics'
SUMMARY_PATH = 'cluster_summary_landscape_vs_infographic.csv'

# Features to test combinations on
SELECTED_FEATURES = [
    'gradient_stats',
    'layout_alignment',
    'color_region_count',
    'lbp',
    'entropy',
    'edge_density',
    'hue_saturation_stats',
    'color_palette_diversity',
    'blur'
]

def load_feature_stack(dir_path, feature_names):
    feature_arrays = []
    for name in feature_names:
        file_path = os.path.join(dir_path, f"{name}.npy")
        arr = np.load(file_path)
        feature_arrays.append(arr)
    return np.concatenate(feature_arrays, axis=1)

def load_labels(label_path, label_prefix):
    labels = np.load(label_path)
    # Here, only label by dataset type: Landscape or Infographic
    # Label 1 or 0 is ignored here, all Landscape labels get "Landscape", all infographic get "Infographic"
    # Because we only want to separate Landscape vs Infographic
    return [label_prefix for _ in labels]

def evaluate_features(feature_combo):
    # Load features
    X_landscape = load_feature_stack(LANDSCAPE_FEAT_DIR, feature_combo)
    y_landscape = load_labels(os.path.join(LANDSCAPE_FEAT_DIR, 'labels.npy'), 'Landscape')

    X_infographic = load_feature_stack(INFOGRAPHIC_FEAT_DIR, feature_combo)
    y_infographic = load_labels(os.path.join(INFOGRAPHIC_FEAT_DIR, 'labels.npy'), 'Infographic')

    X = np.vstack([X_landscape, X_infographic])
    y_true = y_landscape + y_infographic

    # KMeans with 2 clusters to separate Landscape vs Infographic
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X)

    # Map clusters to true labels (Landscape or Infographic) by majority vote
    cluster_to_true_labels = defaultdict(list)
    for cid, true_label in zip(cluster_ids, y_true):
        cluster_to_true_labels[cid].append(true_label)

    cluster_to_label = {
        cid: Counter(labels).most_common(1)[0][0]
        for cid, labels in cluster_to_true_labels.items()
    }

    y_pred = [cluster_to_label[cid] for cid in cluster_ids]

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    ari = adjusted_rand_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Count correct and incorrect classifications by category
    correct_landscape = sum((a == p) and (a == 'Landscape') for a, p in zip(y_true, y_pred))
    incorrect_landscape = sum((a != p) and (a == 'Landscape') for a, p in zip(y_true, y_pred))
    correct_infographic = sum((a == p) and (a == 'Infographic') for a, p in zip(y_true, y_pred))
    incorrect_infographic = sum((a != p) and (a == 'Infographic') for a, p in zip(y_true, y_pred))

    return {
        'features': ', '.join(feature_combo),
        'accuracy': accuracy,
        'f1_score': f1,
        'adjusted_rand_index': ari,
        'balanced_accuracy': balanced_acc,
        'correct_landscape': correct_landscape,
        'incorrect_landscape': incorrect_landscape,
        'correct_infographic': correct_infographic,
        'incorrect_infographic': incorrect_infographic,
    }
