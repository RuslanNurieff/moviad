import numpy as np
import torch
from random import randint


def quantize_vector(vector: np.array, n_subvectors: int, n_centroids: int) -> np.array:
    """
    This function quantizes a given vector into a quantized vector using product quantization
    (https://www.pinecone.io/learn/series/faiss/product-quantization/)
    Args:
        vector: input vector to be quantized
        n_subvectors: number of subvectors for the vector to be divided into (affects the final quantized vector size)
        n_centroids: number of centroids to be created for each subvector (affects the precision of the representation of the vector)
    Returns:
        np.array: quantized vector
    """
    if not isinstance(vector, np.array):
        raise ValueError("Input vector must be a numpy array")
    if not isinstance(n_subvectors, int):
        raise ValueError("Number of subvectors must be an integer")
    if not isinstance(n_centroids, int):
        raise ValueError("Number of centroids must be an integer")

    subvectors, subvector_length = create_subvectors(vector, n_subvectors)
    centroids, k_ = create_centroids(subvector_length, n_subvectors, n_centroids)
    quantized_subvectors = []
    for subvector_index in range(n_subvectors):
        centroid_index = nearest(centroids[subvector_index], subvectors[subvector_index])
        quantized_subvectors.append(centroid_index)
    quantized_vector = np.array(quantized_subvectors).flatten()
    return quantized_vector


def euclidean(v: np.array, u: np.array):
    distance = sum((x - y) ** 2 for x, y in zip(v, u)) ** .5
    return distance


def nearest(c_j: np.array, u_j: np.array) -> int:
    nearest_idx = -1
    distance = 9e9
    for i in range(len(c_j)):
        new_dist = euclidean(c_j[i], u_j)
        if new_dist < distance:
            nearest_idx = i
            distance = new_dist
    return nearest_idx


def create_centroids(subvector_length: int, n_subvectors: int, n_centroids: int) -> (list[np.array], int):
    """
    This function creates centroids for a given list of subvectors

    Args:
        n_subvectors (int): number of subvectors
        n_centroids (int): number of centroids to be created

    Returns:
        np.array: centroids for the given list of subvectors
    """

    n_centroid_per_subvector = int(n_centroids / n_subvectors)

    c = []  # our overall list of reproduction values
    for j in range(n_subvectors):
        # each j represents a subvector (and therefore subquantizer) position
        c_j = []
        for i in range(n_centroid_per_subvector):
            # each i represents a cluster/reproduction value position *inside* each subspace j
            c_ji = [randint(0, 9) for _ in range(subvector_length)]
            c_j.append(c_ji)  # add cluster centroid to subspace list
        # add subspace list of centroids to overall list
        c.append(c_j)
    return c, n_centroid_per_subvector


def create_subvectors(vector: np.array, n_subvectors: int) -> (list[np.array], int):
    """
    This function creates subvectors of a given vector

    Args:
        vector (np.array): vector to be divided into subvectors
        n_subvectors (int): number of subvectors to be created

    Returns:
        np.array: subvectors of the given vector
    """

    vector_length = len(vector)
    subvector_length = int(vector_length / n_subvectors)
    subvector_list = [vector[row:row + subvector_length] for row in range(0, vector_length, subvector_length)]
    return subvector_list, subvector_length
