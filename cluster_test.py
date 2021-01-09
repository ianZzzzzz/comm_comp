
import pandas as pd
from numpy import ndarray

def k_means(clusters:int,array_2d:ndarray):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    plt.figure(figsize=(3, 5))
    random_state = 170
    X = array_2d
    y_pred = KMeans(
        n_clusters=clusters, random_state=random_state
        ).fit_predict(X)
    plt.scatter(
        X[:, 0],
        X[:, 1], 
        c=y_pred,
        alpha = 1,
        s=1.2)
    plt.title("K-MEAN n_clusters=4")

    plt.show()

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

x = np.array([ [7], [8], [9]])
y = np.array([[2], [3], [4]])
distance, path = fastdtw(x, y, dist=euclidean)
print(distance)