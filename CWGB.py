import numpy as np
from sklearn.cluster import KMeans


def CGB_data(data_train, n_clusters):
    """
    对 data_train 数据按照标签进行 KMeans 聚类，并生成聚类结果列表 data_GB。
    当 n_clusters 为 1 时，直接计算中心和半径，不进行聚类。

    :param data_train: 输入数据，二维数组，最后一列为标签。
    :param n_clusters: 每个标签的聚类数量列表，例如 [100, 1]。
    :return: data_GB 列表，每个元素是一个包含中心、半径、标签和原始数据的列表。
    """
    # 按标签分组
    data_by_class = {}
    class_labels = len(n_clusters)
    for label in range(class_labels):
        # 提取标签为 label 的数据，并去掉最后一列标签
        data_by_class[label] = data_train[data_train[:, -1] == label, :-1]

    # 初始化存储结果
    data_GB = []

    def process_cluster(data, n_clusters, label):
        # 当聚类数为1时，直接计算整体中心和半径
        if n_clusters == 1:
            center = np.mean(data, axis=0)
            distances = np.sqrt(np.sum((data - center) ** 2, axis=1))
            radius = np.mean(distances)
            return [[center, radius, label, data]]

        # 否则进行 KMeans 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=7, n_init=2).fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_data = data[labels == cluster_id]
            center = centers[cluster_id]
            radius = np.mean(np.sqrt(np.sum((cluster_data - center) ** 2, axis=1)))
            clusters.append([center, radius, label, cluster_data])
        return clusters

    # 对每个类别进行处理
    for i in range(class_labels):
        data_GB.extend(process_cluster(data_by_class[i], n_clusters[i], label=i))

    return data_GB