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

