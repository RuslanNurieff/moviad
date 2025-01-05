import unittest

import numpy as np

from moviad.utilities.quantization import create_subvectors, create_centroids, nearest

import torch


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        self.original_size = 1027
        self.tensor = torch.rand(self.original_size)

    def test_subvector_creation(self):
        subvectors, D_ = create_subvectors(self.tensor.cpu().numpy(), 4)
        self.assertIsNotNone(subvectors)
        self.assertIsNotNone(D_)

    def test_centroids_creation(self):
        subvectors, D_ = create_subvectors(self.tensor.cpu().numpy(), 4)
        centroids, k_ = create_centroids(subvectors, 4, D_)
        self.assertIsNotNone(centroids)
        self.assertIsNotNone(k_)

    def test_product_quantization(self):
        n_subvectors = 32
        n_centroids = 128
        subvectors, subvector_length = create_subvectors(self.tensor.cpu().numpy(), n_subvectors)
        centroids, k_ = create_centroids(subvector_length, n_subvectors, n_centroids)
        quantized_subvectors = []
        for subvector_index in range(n_subvectors):
            centroid_index = nearest(centroids[subvector_index], subvectors[subvector_index])
            quantized_subvectors.append(centroid_index)

        self.assertIsNotNone(quantized_subvectors)
        self.assertEqual(len(quantized_subvectors), n_subvectors)
        self.assertIsNotNone(centroids)
        self.assertIsNotNone(k_)
        self.assertEqual(len(centroids), n_subvectors)
        self.assertEqual(len(centroids), n_subvectors)
        self.assertEqual(len(centroids[0]), k_)

        quantized_vector = np.array(quantized_subvectors).flatten()
        self.assertIsNotNone(quantized_vector)
        self.assertLess(len(quantized_vector), self.original_size)


if __name__ == '__main__':
    unittest.main()
