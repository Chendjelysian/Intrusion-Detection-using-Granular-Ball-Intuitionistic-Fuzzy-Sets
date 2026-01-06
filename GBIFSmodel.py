import time
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from CWGB import CGB_data
from metric import evaluate_performance, print_meetrics
from preprocess_data import datasetload
from preprocess_data import preprocess_data


def Time_cut(data, t):
    '''
    Data dataset splitting
    '''
    data = np.array(data, dtype=np.float32)
    data_cut = np.array_split(data, t, axis=0)
    data_cut = np.array(data_cut)

    return data_cut


def create_dynamic_if_matrix(samples, labels, n_features, data_mean):
    """Vectorized and optimized function to generate a dynamic Intuitionistic Fuzzy matrix."""
    n_samples = samples.shape[0]
    matrix = np.zeros((n_samples, n_features, 3), dtype=np.float32)

    # --- Core optimization: Vectorized computation of membership components for all samples ---
    # Compute differences between all samples and the feature-wise mean in one go
    diff = samples - data_mean[None, :]  # Shape: (n_samples, n_features)
    abs_diff = np.abs(diff)

    # Batch-compute the three intuitionistic fuzzy components (eliminates per-sample loops)
    membership = (1 - np.sqrt(abs_diff))  # Shape: (n_samples, n_features)
    non_membership = abs_diff
    uncertainty = 1 - membership - non_membership

    # Directly construct the IF matrix using stacking (avoids per-sample concatenation)
    matrix = np.stack([membership, non_membership, uncertainty], axis=2)  # Shape: (n_samples, n_features, 3)

    matrix_mean = np.mean(matrix, axis=0, dtype=np.float32)  # Shape: (n_features, 3)

    # --- Output structure ---
    matrix_list = [
        matrix_mean.copy(),
        np.array(labels[0], dtype=np.int64),
        data_mean.copy()
    ]

    return matrix_list


def optimized_process(B_batch, X_avg_matrix, matrix_array, t):
    """
    Vectorized computation of matching scores for a batch of test samples.

    Args:
        B_batch: Current batch of test samples, shape (batch_size, 30)
        X_avg_matrix: Centers of all granular balls, shape (GB_num, t, 30)
        matrix_array: Intuitionistic fuzzy components of all granular balls, shape (GB_num, t, 30, 3)
        t: Number of time steps

    Returns:
        scores: Matching score matrix, shape (batch_size, GB_num)
    """
    # Initialize weighting parameters

    dim1 = 1  # exponent for distance weighting

    # Optional: print parameter settings (currently commented out)
    # print(f"Membership degree scaled by factor: {men}")
    # print(f"Non-membership degree scaled by factor: {nonmen}")
    # print(f"Hesitation degree scaled by factor: {hes}")
    # print(f"Distance metric raised to the power of: {dim1}")
    # print(f"Additional weighting exponent: {dim2}")

    # Expand dimensions to enable broadcasting
    # B_batch: (batch_size, 1, 1, n_features)
    # X_avg_matrix: (1, GB_num, t, n_features)
    diff = B_batch[:, None, None, :] - X_avg_matrix[None, :, :, :]  # (batch, GB, t, n_features)
    abs_diff = np.abs(diff)

    # Compute intuitionistic fuzzy components using broadcasting
    membership = (1 - np.sqrt(abs_diff))  # (batch, GB, t, n_features)
    non_membership = abs_diff  # (batch, GB, t, n_features)
    uncertainty = (1 - membership - non_membership)  # (batch, GB, t, n_features)

    # Extract reference fuzzy components from granular balls
    # matrix_array: (GB_num, t, 30, 3) → split into three channels
    ref_mu = matrix_array[:, :, :, 0][None, :, :, :]  # (1, GB, t, n_features)
    ref_nu = matrix_array[:, :, :, 1][None, :, :, :]  # (1, GB, t, n_features)
    ref_pi = matrix_array[:, :, :, 2][None, :, :, :]  # (1, GB, t, n_features)

    # Compute three distance components with aligned dimensions
    d_mu = np.abs(membership - ref_mu) ** dim1 / (1 + membership + ref_mu)  # (batch, GB, t, n_features)
    d_nu = np.abs(non_membership - ref_nu) ** dim1 / (1 + non_membership + ref_nu)
    d_pi = np.abs(uncertainty - ref_pi) ** dim1 / (1 + uncertainty + ref_pi)

    # Average over time and feature dimensions to obtain final scores
    scores = np.mean(d_mu * men + d_nu * nomen + d_pi * hes, axis=(2, 3))  # (batch_size, GB_num)
    return scores


def create_dynamic_if_matrix_test(test_samples, n_features, matrix_list_GB, t):
    """
    Constructs a dynamic Intuitionistic Fuzzy Set (IFS) matrix for test samples,
    using batch processing to handle large datasets efficiently.

    Args:
        test_samples: Test data array of shape (n_samples, n_features)
        n_features: Number of features per sample
        matrix_list_GB: List of granular balls, where each entry is a list over time steps.
                        Each time step contains [IFS_matrix (n_features, 3), label (int), center (n_features,)]
        t: Number of time steps per granular ball

    Returns:
        results: Matching scores array of shape (n_samples, GB_num), where GB_num is the number of granular balls
    """
    n_samples = test_samples.shape[0]
    GB_num = len(matrix_list_GB)
    t = len(matrix_list_GB[0])  # Number of time steps per granular ball

    # Initialize storage arrays
    matrix_array = np.zeros((GB_num, t, n_features, 3),
                            dtype=np.float32)  # Stores [membership, non-membership, hesitation]
    X_avg_matrix = np.zeros((GB_num, t, n_features),
                            dtype=np.float32)  # Stores centroid (x_avg) of each granular ball at each time step
    labels = np.zeros((GB_num, t), dtype=int)  # Stores labels of granular balls

    # Populate arrays from matrix_list_GB
    for GB_idx in range(GB_num):
        for t_step in range(t):
            # matrix_list_GB[GB_idx][t_step] = [ (n_features, 3) IFS matrix, scalar label, (n_features,) centroid ]
            matrix_array[GB_idx, t_step] = matrix_list_GB[GB_idx][t_step][0]  # (n_features, 3)
            labels[GB_idx, t_step] = matrix_list_GB[GB_idx][t_step][1]  # scalar label
            X_avg_matrix[GB_idx, t_step] = matrix_list_GB[GB_idx][t_step][2]  # (n_features,)

    # Process test samples in batches to manage memory usage
    batch_size = 300  # Adjust based on available memory
    results = []

    for i in range(0, n_samples, batch_size):
        B_batch = test_samples[i:i + batch_size]  # Shape: (batch_size, n_features)
        batch_scores = optimized_process(B_batch, X_avg_matrix, matrix_array, t)
        results.append(batch_scores)

    # Combine batch results and return
    results = np.vstack(results)  # Shape: (n_samples, GB_num)
    results = np.array(results, dtype=np.float32)
    return results


def classifier(average_results, matrix_list_GB):
    """
    Classifies test samples based on their matching scores with granular balls
    using a k-nearest neighbor voting strategy.

    Args:
        average_results: Matching score matrix of shape (n_samples, GB_num),
                         where lower scores indicate higher similarity.
        matrix_list_GB: List of granular balls used during training;
                        needed to retrieve class labels.

    Returns:
        predicted_labels: List of predicted class labels for each test sample.
    """
    results_with_t = average_results
    predicted_labels = []
    k = 1  # Number of nearest neighbors to consider (k=1 for 1-NN)

    # Pre-extract labels of all granular balls (using label at time step 0 as representative)
    labels_all = [matrix_list_GB[idx][0][1] for idx in range(len(matrix_list_GB))]
    labels_all = np.array(labels_all, dtype=int)

    for arr in results_with_t:
        # Find indices of the k smallest scores (most similar granular balls)
        min_indices = np.argpartition(arr, k, axis=None)[:k]

        # Retrieve corresponding labels
        neighbor_labels = labels_all[min_indices]

        # Majority vote among k neighbors (for k=1, this is just the single label)
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        # Alternative (faster for small k): most_common_label = np.bincount(neighbor_labels).argmax()

        predicted_labels.append(most_common_label)

    return predicted_labels


def main(random_state, data_x, data_y, labels_len, GB_num, feature, importances=None):
    feature_seletion = 0

    features_nums = np.zeros(120)
    t = 1
    a = 0
    data_train1, data_test, labels_train, labels_test = preprocess_data(data_x, data_y, random_state)

    for j in range(feature, feature + 1):
        features_nums[a] = j

        if feature_seletion == 0:

            model_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            model_rf.fit(data_train1, labels_train)

            # Obtain feature importance scores
            importances = model_rf.feature_importances_

            k = j
            beginfeature = k
            sorted_indices = np.argsort(importances)[-k:]
            data_x = np.array(data_x)
            # Filter the top k features
            data_x1 = data_train1[:, sorted_indices]
            data_test1 = data_test[:, sorted_indices]
            print("Number of features used", sorted_indices)
            feature_seletion = 1
        else:

            k = j
            print("Number of features used:", k)
            sorted_indices = np.argsort(importances)[-k:]
            data_x = np.array(data_x)
            # Filter the top k features
            data_x1 = data_train1[:, sorted_indices]
            data_test1 = data_test[:, sorted_indices]
            print("Feature index used", sorted_indices)

        n_features = data_x1.shape[1]
        data_train = np.concatenate((data_x1, labels_train.reshape(-1, 1)), axis=1)

        training_timelen = len(data_train)
        # GB generation
        start_time = time.time()
        Generation_method = 2
        if Generation_method == 2:
            nGB = GB_num
            n_clusters = []
            for _ in range(labels_len):
                n_clusters.append(nGB)
            # n_clusters[0] = 200
            # n_clusters[1] = 200
            data_GB = CGB_data(data_train, n_clusters)
            print("Number of pellets in each category", n_clusters)

        matrix_list_GB = []  #
        for data_gb in data_GB:  # Each of the n lists contains: central mean, radius, label, and data.
            data_mean = data_gb[0]  # Center
            data_R = data_gb[1]  # radius
            data_l = data_gb[2]  # label
            data_origin = data_gb[3]  # data

            train_lens, n_features = data_origin.shape[0], data_origin.shape[1]
            # Dynamically cut by time but t=1, just no use
            data_train = Time_cut(data_origin, t)
            # Use NumPy arrays instead of lists
            labels_train1 = np.full(2, data_l, dtype=type(data_l))  # The specified data type is data_l
            labels_train1 = Time_cut(labels_train1, t)

            # Creating intuitionistic fuzzy matrices
            matrix_list_all = []

            for i in range(t):
                matrix_list = create_dynamic_if_matrix(
                    data_train[i], labels_train1[i], n_features, data_mean)

                matrix_list_all.append(matrix_list)

            matrix_list_GB.append(matrix_list_all)

        end_time = time.time()
        training_time = end_time - start_time
        print("Training time：", training_time)
        print("Training time/training set：", training_time / training_timelen)

        # 测试集
        start_time = time.time()
        average_results = create_dynamic_if_matrix_test(data_test1, n_features, matrix_list_GB, t)
        end_time = time.time()
        test_matrix_time = end_time - start_time
        print("Test set build time：", test_matrix_time)
        print("Test set construction time / Test set：", test_matrix_time / len(labels_test))

        # 预测
        start_time = time.time()
        predictions_indices = classifier(average_results, matrix_list_GB)  # 换成粒球输入
        end_time = time.time()
        pre_time = end_time - start_time
        print("Predicted time：", pre_time)
        print("Prediction time/prediction set：", pre_time / len(predictions_indices))

        # 评估性能
        accuracy, macro_precision, macro_recall, macro_f1_score, results = evaluate_performance(
            np.array(predictions_indices),
            labels_test)

        Accuracy, Precision, Recall, F1_score, C1_Recall, C1_F1_score, FPR = print_meetrics(a, accuracy,
                                                                                            macro_precision,
                                                                                            macro_recall,
                                                                                            macro_f1_score, results)
        print('The number of features selected is:', n_features)
        a = a + 1

    return Accuracy, Precision, Recall, F1_score, C1_Recall, C1_F1_score, FPR, beginfeature, features_nums


if __name__ == '__main__':
    random_states = [7, 17, 42]
    Accuracy3 = []
    macro_precision3 = []
    macro_recall3 = []
    macro_f1_score3 = []
    results3 = []
    C1_Recall3 = []
    C1_F1_score3 = []
    FPR3 = []
    #
    global men  # weight for membership degree
    global nomen  # weight for non-membership degree
    global hes  # weight for hesitation (uncertainty) degree
    men = 1
    nomen = 1
    hes = 10

    # filename = r"data\X-IIOTID\X-IIoTID dataset.csv"
    # GB_num = 100
    # feature = 10

    # filename = r"data\TON-IOT\Train_Test_Network.csv"
    # GB_num = 5
    # feature = 14

    # filename = r"data\WUSTL-IIOT\wustl_iiot_2021.csv"
    # GB_num = 10
    # feature = 13

    filename = r"data\KDDCUP99\kddcup.data"
    GB_num = 25
    feature = 37

    # filename = r"data\UNSW-NB15\UNSW-NB15_full.csv"
    # GB_num = 20
    # feature = 6

    # filename = r"data\NSLKDD\KDDTest+.csv"
    # GB_num = 400
    # feature = 23

    data_x, data_y, labels_len = datasetload(filename)

    for random_state in random_states:
        Accuracy, macro_precision, macro_recall, macro_f1_score, C1_Recall, C1_F1_score, FPR, beginfeature, \
            features_nums = main(random_state, data_x, data_y, labels_len, GB_num, feature)

        Accuracy3.append(Accuracy)
        macro_precision3.append(macro_precision)
        macro_recall3.append(macro_recall)
        macro_f1_score3.append(macro_f1_score)
        C1_Recall3.append(C1_Recall)
        C1_F1_score3.append(C1_F1_score)
        FPR3.append(FPR)

        # Accuracy里的最大值-某个特征下
        max_accuracy = max(Accuracy)
        # 最大值的索引
        max_index = np.argmax(Accuracy)
        max_index += beginfeature
        print('The maximum accuracy for each feature is:', max_accuracy)
        print("The number of features selected is:", max_index)
        non_zero_accuracy = Accuracy[Accuracy != 0]
        print('The accuracy rate list is:', non_zero_accuracy)

        print('------------------')
        print("\n")


    def datemean(resultstemp3, max_index):

        resultstemp = np.mean(resultstemp3, axis=0)
        max_resultstemp = resultstemp[max_index]

        non_zero_resultstemp = resultstemp[resultstemp != 0]
        return resultstemp, max_resultstemp, non_zero_resultstemp


    print("3 average results")
    Accuracy = np.mean(Accuracy3,
                       axis=0)  # Averaging of multiple random numbers (finding the maximum average across all features)
    max_accuracy = max(Accuracy)
    max_index = np.argmax(Accuracy)
    max_indextemp = beginfeature + max_index
    non_zero_accuracy = Accuracy[Accuracy != 0]
    print('Maximum average accuracy is:', max_accuracy)
    print("The number of features corresponding to the maximum average accuracy is:", max_indextemp)
    print('The average accuracy list is as follows:', non_zero_accuracy)
    print('------------------')
    macro_precision, max_macro_precision, non_zero_macro_precision = datemean(macro_precision3, max_index)
    macro_recall, max_macro_recall, non_zero_macro_recall = datemean(macro_recall3, max_index)
    macro_f1_score, max_f1_score, non_zero_f1_score = datemean(macro_f1_score3, max_index)
    # C1_Recall, max_C1_Recall, non_zero_C1_Recall = datemean(C1_Recall3, max_index)
    # C1_F1_score, max_C1_F1_score, non_zero_C1_F1_score = datemean(C1_F1_score3, max_index)
    FPR, max_FPR, non_zero_FPR = datemean(FPR3, max_index)
    print('The macro_precision corresponding to the highest average accuracy is:', max_macro_precision)
    print('The macro_recall corresponding to the highest average accuracy is:', max_macro_recall)
    print('The macro_f1_score corresponding to the highest average accuracy is:', max_f1_score)
    # print('The C1_Recall corresponding to the highest average accuracy is:', max_C1_Recall)
    # print('The C1_F1_score corresponding to the highest average accuracy is:', max_C1_F1_score)
    print('The FPR corresponding to the highest average accuracy is:', max_FPR)
    print('List of macro_precision values:', non_zero_macro_precision)
    print('List of macro_recall values:', non_zero_macro_recall)
    print('List of macro_f1_score values:', non_zero_f1_score)
    # print('List of C1_Recall values:', non_zero_C1_Recall)
    # print('List of C1_F1_score values:', non_zero_C1_F1_score)
    print('List of FPR values:', non_zero_FPR)
    print('------------------')
