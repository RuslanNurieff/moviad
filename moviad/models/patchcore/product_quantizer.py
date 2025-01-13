import faiss
import numpy as np
import torch

class ProductQuantizer:
    quantizer: faiss.IndexPQ
    dim = 1
    subspaces: int
    centroids_per_subspace: int = 256


    def fit(self, input: torch.Tensor | np.ndarray, dim=1) -> None:
        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()
        self.dim = dim
        self.subspaces = self.__compute_optimal_m(input)

        self.quantizer = faiss.IndexPQ(input.shape[dim], self.subspaces, int(np.log2(self.centroids_per_subspace)))
        self.quantizer.train(input)
        self.quantizer.add(input)

    def encode(self, input: torch.Tensor | np.ndarray, dim = 0) -> torch.Tensor:
        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()

        compressed = np.zeros((input.shape[dim], self.subspaces), dtype=np.uint8)

        self.quantizer.sa_encode(input, compressed)

        return torch.tensor(compressed, dtype=torch.float32)

    def decode(self, input: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()

        if input.dtype != np.uint8:
            input = input.astype(np.uint8)

        decompressed = np.zeros((input.shape[0], self.quantizer.d), dtype=np.float32)
        self.quantizer.sa_decode(input, decompressed)
        return torch.tensor(decompressed, dtype=torch.float32)

    def __compute_optimal_m(self, input: np.ndarray) -> int:
        d = input.shape[self.dim]

        # Find all divisors of d
        divisors = [m for m in range(1, d + 1) if d % m == 0]

        # Filter based on subvector dimensionality constraints
        valid_m = [m for m in divisors]

        # Suggest an optimal m (default to 8 if valid, else largest valid value)
        suggested_m = 8 if 8 in valid_m else min(valid_m)

        return suggested_m