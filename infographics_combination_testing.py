import os
import numpy as np
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import csv


# List of features to test combinations with
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

# Pathway to where all feature files are stored
FEATURE_DIR = 'features_infographics'

# Output CSV path for saving results
RESULTS_CSV_PATH = 'infographic_feature_combination_results.csv'

# Load feature vectors and labels

print("Loading feature data...")
feature_matrix_by_name = {}
for feature_name in SELECTED_FEATURES:
    file_path = os.path.join(FEATURE_DIR, f'{feature_name}.npy')
    feature_matrix_by_name[feature_name] = np.load(file_path)

# Load labels, 0 = AI-generated, 1 = human-made
labels = np.load(os.path.join(FEATURE_DIR, 'labels.npy'))
results_summary = []

# Combination testing

print("Exploring all possible combinations of selected features...\n")
for num_features in range(1, len(SELECTED_FEATURES) + 1):
    for feature_combo in itertools.combinations(SELECTED_FEATURES, num_features):
        
        # Combine selected feature arrays side-by-side
        combined_features = np.concatenate([feature_matrix_by_name[name] for name in feature_combo], axis=1)
        
        # Normalize each feature dimension
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(combined_features)

        # Dimensionality reduction using PCA (retain 95% of variance)
        pca = PCA(n_components=0.95)
        reduced_features = pca.fit_transform(normalized_features)

        # Apply KMeans clustering to split data into two groups
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(reduced_features)

        # Manually map cluster labels (0/1) to match class labels
        cluster_0 = [0, 0] 
        cluster_1 = [0, 0] 
        for i, label in enumerate(labels):
            if cluster_assignments[i] == 0:
                cluster_0[label] += 1
            else:
                cluster_1[label] += 1

        # Choose cluster-to-class mapping that maximizes correct matches
        if cluster_0[0] + cluster_1[1] > cluster_1[0] + cluster_0[1]:
            cluster_to_class = {0: 0, 1: 1}
        else:
            cluster_to_class = {0: 1, 1: 0}

        # Map predicted clusters to class labels
        predicted_labels = np.array([cluster_to_class[c] for c in cluster_assignments])

        # Evaluation metrics

        accuracy = np.mean(predicted_labels == labels)
        f1 = f1_score(labels, predicted_labels)
        ari = adjusted_rand_score(labels, predicted_labels)

        # Breakdown of prediction performance
        correct_ai = np.sum((labels == 0) & (predicted_labels == 0))
        correct_human = np.sum((labels == 1) & (predicted_labels == 1))
        incorrect_ai = np.sum((labels == 0) & (predicted_labels == 1))
        incorrect_human = np.sum((labels == 1) & (predicted_labels == 0))

        recall_ai = correct_ai / np.sum(labels == 0)
        recall_human = correct_human / np.sum(labels == 1)
        balanced_accuracy = 0.5 * (recall_ai + recall_human)

        # Store results for this feature combo
        results_summary.append({
            'features': ', '.join(feature_combo),
            'accuracy': accuracy,
            'f1_score': f1,
            'adjusted_rand_index': ari,
            'balanced_accuracy': balanced_accuracy,
            'correct_ai': correct_ai,
            'correct_real': correct_human,
            'incorrect_ai': incorrect_ai,
            'incorrect_real': incorrect_human
        })

# Sort all combinations by balanced accuracy
results_summary.sort(key=lambda x: x['balanced_accuracy'], reverse=True)

# Save all results to a CSV file
with open(RESULTS_CSV_PATH, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=[
        'features', 'accuracy', 'f1_score', 'adjusted_rand_index',
        'balanced_accuracy', 'correct_ai', 'correct_real',
        'incorrect_ai', 'incorrect_real'
    ])
    writer.writeheader()
    writer.writerows(results_summary)

print(f"All combinations evaluated and results saved to → {RESULTS_CSV_PATH}")
print(f"Best combination: {results_summary[0]['features']} (Balanced Acc: {results_summary[0]['balanced_accuracy']:.4f})")