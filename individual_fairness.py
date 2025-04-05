from sklearn.neighbors import NearestNeighbors
import numpy as np
import gower

def eval_ind_fairness(x_train, y_train, x_test, y_pred):
    # Compute Gower distance matrix for test samples w.r.t training data
    gower_distances = gower.gower_matrix(x_test, x_train)  # Shape: (num_test_samples, num_train_samples)

    # Find k nearest neighbors (excluding self)
    k = 5  # Adjust as needed
    neighbors = np.argsort(gower_distances, axis=1)[:, 1:k+1]  # Get indices of k nearest neighbors

    # Compute consistency score: Fraction of nearest neighbors with same prediction
    consistencies = []
    for i, neigh_indices in enumerate(neighbors):
        neighbor_preds = y_train.iloc[neigh_indices]  # Get predictions of k neighbors from training labels
        consistency = np.mean(neighbor_preds == y_pred[i])  # Fraction with same prediction
        consistencies.append(consistency)

    # Calculate overall consistency score
    individual_fairness_score = np.mean(consistencies)

    return individual_fairness_score