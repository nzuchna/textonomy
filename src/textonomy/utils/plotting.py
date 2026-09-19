import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.cluster import contingency_matrix

def plot_cluster_confusion_matrix(model, labels_true, labels_pred, save_path="../results/tmp/cluster_confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    unique_labels1 = np.unique(labels_true)
    unique_labels2 = np.unique(labels_pred)

    # decode labels
    unique_labels_true = unique_labels1
    unique_labels_pred = model.label_encoder.inverse_transform(unique_labels2)

    # Compute contingency matrix
    cm = contingency_matrix(labels_true=labels_true, labels_pred=labels_pred)

    # Compute row and column sums
    row_sums = cm.sum(axis=1, keepdims=True)  # Sum of each row
    col_sums = cm.sum(axis=0, keepdims=True)  # Sum of each column
    total_sum = np.array([[cm.sum()]])        # Grand total

    # Extend the matrix with row and column sums
    cm_extended = np.vstack([cm, col_sums])   # Add row sums at the bottom
    cm_extended = np.hstack([cm_extended, np.vstack([row_sums, total_sum])])  # Add col sums on the right

    # Extend labels with "Total"
    unique_labels_true_extended = np.append(unique_labels_true, "Total")
    unique_labels_pred_extended = np.append(unique_labels_pred, "Total")

    # Plot heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_extended, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels_pred_extended, yticklabels=unique_labels_true_extended)
    plt.xlabel("Clustering 2 (Predicted Labels)")
    plt.ylabel("Clustering 1 (True Labels)")
    plt.title("Clustering Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
