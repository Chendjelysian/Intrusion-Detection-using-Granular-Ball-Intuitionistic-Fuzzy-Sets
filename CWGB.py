import numpy as np
from sklearn.cluster import KMeans


def CGB_data(data_train, n_clusters):
    """
    Performs class-wise KMeans clustering on `data_train` and constructs a list of granular balls (`data_GB`).
    If `n_clusters` is 1 for a given class, the entire class data is treated as a single granular ball,
    with its center and radius computed directly without clustering.

    Args:
        data_train: Input training data as a 2D array, where the last column contains class labels.
        n_clusters: A list specifying the number of clusters to form for each class.
                    For example, [100, 1] means 100 clusters for class 0 and 1 cluster for class 1.

    Returns:
        data_GB: A list of granular balls. Each element is a list of the form:
                 [center (1D array), radius (float), label (int), original_data (2D array)]
    """
    # Group data by class label
    data_by_class = {}
    num_classes = len(n_clusters)
    for label in range(num_classes):
        # Extract samples belonging to the current class and remove the label column
        data_by_class[label] = data_train[data_train[:, -1] == label, :-1]

    # Initialize the output list
    data_GB = []

    def process_cluster(data, n_clusters, label):
        """
        Processes a single class: either forms one granular ball or applies KMeans clustering.
        """
        # Case: no clustering — treat all samples as one granular ball
        if n_clusters == 1:
            center = np.mean(data, axis=0)
            distances = np.sqrt(np.sum((data - center) ** 2, axis=1))
            radius = np.mean(distances)
            return [[center, radius, label, data]]

        # Case: perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=7, n_init=10).fit(data)
        centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_data = data[cluster_labels == cluster_id]
            center = centers[cluster_id]
            # Compute average Euclidean distance from center as the radius
            radius = np.mean(np.sqrt(np.sum((cluster_data - center) ** 2, axis=1)))
            clusters.append([center, radius, label, cluster_data])
        return clusters

    # Process each class according to its specified number of clusters
    for class_idx in range(num_classes):
        data_GB.extend(process_cluster(data_by_class[class_idx], n_clusters[class_idx], label=class_idx))

    return data_GB
