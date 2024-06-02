import random
import numpy as np
import torch


def set_seeds(seed: int=42):

    # Python RNG
    random.seed(seed)

    # Numpy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def eigen_centrality(matrix):

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Eigenvector corresponding to the maximum eigenvalue
    max_index = np.argmax(eigenvalues)

    return abs(eigenvectors[:, max_index])


def calculate_mi(rho):
    if 1 - rho**2 <= 0:
        return float('inf')  # Avoid log of zero or negative
    return -0.5 * np.log(1 - rho**2)


def generate_correlation(n_feat=10, diag_strength=10, W_rank=2, W_relative=4):

    set_seeds()

    # Number of variables
    W_dist = 45 * np.arange(0,n_feat)**W_relative / (np.arange(0,n_feat)**W_relative).sum()
    W_strength = np.tile(W_dist, [W_rank,1]).T

    # Create a diagonal matrix with positive variance terms
    variances = np.abs(np.random.normal(diag_strength, 1, n_feat))
    diag = np.diag(variances)
    W = np.abs(np.random.normal(loc=W_strength, scale=3, size=(n_feat,W_rank)))

    # Create a full covariance matrix by taking the outer product of two vectors to create a low rank matrix then add a diagonal
    covariance = np.matmul(W, W.T) + diag

    # Create a normalizer by taking the multiplicative inverse of the square root of the diagonal
    normalizer = np.diag(1 / np.sqrt(np.diag(covariance)))

    # Create the correlation matrix by normalizing the covariance matrix
    correlation = normalizer @ covariance @ normalizer

    return correlation


def generate_features(n_samples, n_feat, correlation=np.array([]), sparsity=False, sparse=0, outliers=False, num_outliers=100):

    set_seeds()

    if correlation.size == 0:
        correlation = np.eye(n_feat)

    # Simulate
    features_linear = np.random.multivariate_normal(np.zeros(n_feat), correlation, n_samples)

    # Sparsify
    if sparsity:
        features_linear[:,sparse] = (np.random.rand(n_samples) < 0.50).astype(float) * features_linear[:,sparse]

    # Randomize outliers
    if outliers:
        outlier_indices = np.random.choice(n_samples, num_outliers, replace=False)

    return features_linear


# def generate_from_linear(n_samples, n_feat, features, outlier_indices, outlier_feature=4, outlier_interaction=5):

#     set_seeds()

#     # Set up coefficents: ones for now
#     coefficients = np.ones(n_feat).reshape(n_feat,1)

#     # Create noise
#     noise = np.random.normal(0, 1, n_samples).reshape(n_samples,1)

#     # Simulate (linear) generative process
#     y_linear_from_linear = features @ coefficients + noise.reshape(n_samples,1)

#     # Add outliers effects
#     y_linear_from_linear[outlier_indices] += 1000*features[outlier_indices,outlier_feature].reshape(num_outliers,1)


#     ######


#     # Random funcs for nonlinear target y
#     random_func_indices = np.random.randint(low=0, high=len(nonlinear_transformations), size=n_feat//2)

#     # Nonlinear from linear
#     y_nonlinear_from_linear = np.zeros(n_samples).reshape(n_samples,1)

#     # Simulate (nonlinear) generative process
#     for j in range(0,features.shape[1],2):
#         feature_j = features[:,j].reshape(n_samples,1)
#         feature_k = features[:,j+1].reshape(n_samples,1)
#         random_func = nonlinear_transformations[random_func_indices[j//2]]
#         if j == outlier_feature:
#             y_nonlinear_from_linear = y_nonlinear_from_linear + 10*random_func(feature_j*feature_k)
#         else:
#             y_nonlinear_from_linear = y_nonlinear_from_linear + random_func(feature_j*feature_k)

#     # Add outliers effects
#     y_nonlinear_from_linear[outlier_indices] += 1000*(features[outlier_indices,outlier_feature] * features[outlier_indices,outlier_interaction]).reshape(num_outliers,1)

#     return y_linear_from_linear, y_nonlinear_from_linear


# def generate_from_nonlinear(features, outlier_indices, outlier_feature=4, outlier_interaction=5):

#     set_seeds()

#     # Prevent overwriting
#     features_nonlinear = features.copy()

#     # Random funcs for nonlinear features
#     random_func_indices = np.random.randint(low=0, high=len(nonlinear_transformations), size=n_feat)

#     # Make features nonlinear
#     xvec = np.linspace(-4,4,10000)
    
#     for i in range(n_feat-1):
#         random_func = nonlinear_transformations[random_func_indices[i]]
#         features_nonlinear[:,i] = features_linear[:,i]*features_linear[:,i+1] + random_func(xvec)

#     # Linear from nonlinear
#     y_linear_from_nonlinear = features_nonlinear @ coefficients + noise.reshape(n_samples,1)

#     # Add outliers effects
#     y_linear_from_nonlinear[outlier_indices] += 1000*features[outlier_indices,outlier_feature].reshape(num_outliers,1)


#     ######


#     # Random funcs for nonlinear target y
#     random_func_indices2 = np.random.randint(low=0, high=len(nonlinear_transformations), size=n_feat//2)

#     # Nonlinear from nonlinear
#     y_nonlinear_from_linear = np.zeros(n_samples).reshape(n_samples,1)

#     # Simulate (nonlinear) generative process
#     for j in range(0,features_linear.shape[1],2):
#         feature_j = features_linear[:,j].reshape(n_samples,1)
#         feature_k = features_linear[:,j+1].reshape(n_samples,1)
#         random_func = nonlinear_transformations[random_func_indices2[j//2]]
#         if j == 4:
#             y_nonlinear_from_linear = y_nonlinear_from_linear + 10*random_func(feature_j*feature_k)
#         else:
#             y_nonlinear_from_linear = y_nonlinear_from_linear + random_func(feature_j*feature_k)

#     # Add outliers effects
#     y_nonlinear_from_nonlinear[outlier_indices] += 1000*(features_nonlinear[outlier_indices,outlier_feature] * features_nonlinear[outlier_indices,outlier_interaction]).reshape(num_outliers,1)

#     return features_nonlinear, y_linear_from_nonlinear, y_nonlinear_from_nonlinear