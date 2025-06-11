import torch
from src.torch_kmeans.clustering import KMeans

model = KMeans(n_clusters=4)

x = torch.randn((4, 20, 2))   # (BS, N, D)
result = model(x)
print(result.labels)