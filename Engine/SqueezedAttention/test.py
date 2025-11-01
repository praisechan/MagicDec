# Both import methods supported
# import torch
from cuml import KMeans
from cuml.cluster import KMeans
import cudf
import numpy as np
import pandas as pd
a = np.asarray([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]],dtype=np.float32)
b = cudf.DataFrame(a)
# Calling fit
kmeans_float = KMeans(n_clusters=2, n_init="auto")
kmeans_float.fit(b)
# Labels:
kmeans_float.labels_
# cluster_centers:
print(kmeans_float.cluster_centers_)
