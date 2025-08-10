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