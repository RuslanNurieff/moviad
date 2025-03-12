import torch
from scipy.cluster.vq import kmeans, kmeans2
from sklearn.cluster import MiniBatchKMeans
from typing_extensions import override

from moviad.models.patchcore.kcenter_greedy import CoresetExtractor


class KMeansCoresetExtractor(CoresetExtractor):

    def get_coreset_idx_randomp(self, z_lib, n: int = 1000, k: int = 30000, eps: float = 0.90, float16: bool = True,
                                force_cpu: bool = False):
        super().get_coreset_idx_randomp(z_lib, n, k, eps, float16, force_cpu)

    def __init__(self, quantized, device: torch.device, sampling_ratio: float = 0.1,
                 k: int = 30000) -> None:
        super().__init__(quantized, device, sampling_ratio, k)

    def extract_coreset(self, embeddings: torch.Tensor)  -> torch.Tensor:
        batch_size = 256 # TODO: Needs an optimal value
        kmeans_extractor = MiniBatchKMeans(n_clusters=self.k, random_state=42, batch_size=batch_size, n_init="auto")
        kmeans_extractor.fit(embeddings.cpu().numpy())
        coreset = kmeans_extractor.cluster_centers_
        return torch.tensor(coreset).to(self.device)