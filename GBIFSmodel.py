import sys
import time
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from CWGB import CGB_data
from metric import evaluate_performance,print_meetrics
from preprocess_data import datasetload
from preprocess_data import preprocess_data



def Time_cut(data, t):
    '''
    data数据集进行切割
    '''
    data = np.array(data, dtype=np.float32)
    data_cut = np.array_split(data, t, axis=0)
    data_cut = np.array(data_cut)  # 直接切可能用处不大，需要新的方法，至少切完不能平均

    return data_cut


def create_dynamic_if_matrix(samples, labels, n_features, data_mean):
    """向量化优化的动态IF矩阵生成函数"""
    n_samples = samples.shape[0]
    matrix = np.zeros((n_samples, n_features, 3), dtype=np.float32)

    # --- 核心优化：向量化计算所有样本的隶属度分量 ---
    # 一次性计算所有样本与均值的差异
    diff = samples - data_mean[None, :]  # (n_samples, n_features)
    abs_diff = np.abs(diff)

    # 批量计算三个分量（消除逐样本循环）
    membership = (1 - np.sqrt(abs_diff))  # (n_samples, n_features)
    non_membership = abs_diff
    uncertainty = 1 - membership - non_membership

    # 直接构造矩阵（避免逐样本拼接）
    matrix = np.stack([membership, non_membership, uncertainty], axis=2)  # (n_samples, n_features, 3)

    matrix_mean = np.mean(matrix, axis=0, dtype=np.float32)  # (n_features, 3)

    # --- 兼容原输出结构 ---
    matrix_list = [
        matrix_mean.copy(),
        np.array(labels[0], dtype=np.int64),
        data_mean.copy()
    ]

    return matrix_list


def optimized_process(B_batch, X_avg_matrix, matrix_array, t):
    """
    向量化计算批次样本的匹配分数
    Args:
        B_batch: 当前批次的测试样本 (batch_size, 30)
        X_avg_matrix: 所有粒球的中心 (GB_num, t, 30)
        matrix_array: 所有粒球的模糊分量 (GB_num, t, 30, 3)
        t: 时间步数
    Returns:
        scores: 匹配分数矩阵 (batch_size, GB_num)
    """
    # 参数初始化
    men = 1
    nonmen = 1
    hes = 1
    dim1 = 1
    dim2 = 1

    # 打印参数信息
    # print(f"现在是{men}倍隶属度")
    # print(f"现在是{nonmen}倍非隶属度")
    # print(f"现在是{hes}倍犹豫度")
    # print(f"对初步度量加了{dim1}次方权重")
    # print(f"现在是{dim2}次方权重")

    # 扩展维度实现广播计算
    # B_batch: (batch_size, 1, 1, n_features)
    # X_avg_matrix: (1, GB_num, t, n_features)
    diff = B_batch[:, None, None, :] - X_avg_matrix[None, :, :, :]  # (batch, GB, t, n_features)
    abs_diff = np.abs(diff)

    # 计算模糊分量 (利用广播)
    membership = (1 - np.sqrt(abs_diff)) * men  # (batch, GB, t, n_features)
    non_membership = abs_diff * nonmen  # (batch, GB, t, n_features)
    uncertainty = (1 - membership - non_membership) * hes  # (batch, GB, t, n_features)

    # 提取粒球的参考模糊分量 (GB_num, t, 30, 3) -> 广播到 (batch, GB, t, n_features, 3)
    ref_mu = matrix_array[:, :, :, 0][None, :, :, :]  # (1, GB, t, n_features)
    ref_nu = matrix_array[:, :, :, 1][None, :, :, :]  # (1, GB, t, n_features)
    ref_pi = matrix_array[:, :, :, 2][None, :, :, :]  # (1, GB, t, n_features)

    # 计算三个距离分量（维度对齐）
    d_mu = np.abs(membership - ref_mu) ** dim1 / (1 + membership + ref_mu)  # (batch, GB, t, n_features)
    d_nu = np.abs(non_membership - ref_nu) ** dim1 / (1 + non_membership + ref_nu)
    d_pi = np.abs(uncertainty - ref_pi) ** dim1 / (1 + uncertainty + ref_pi)

    # 沿特征和时间维度求平均
    scores = np.mean(d_mu + d_nu + d_pi, axis=(2, 3))  # (batch_size, GB_num)
    return scores


def create_dynamic_if_matrix_test(test_samples, n_features, matrix_list_GB, t):
    """
    创建动态 IFS 矩阵测试函数，使用并行化对样本索引进行处理
    """
    n_samples = test_samples.shape[0]

    # 转换为 NumPy 数组
    GB_num = len(matrix_list_GB)
    t = len(matrix_list_GB[0])  # 每个粒球的时间步数

    # 初始化存储结构
    matrix_array = np.zeros((GB_num, t, n_features, 3), dtype=np.float32)  # 存储隶属度、非隶属度、犹豫度
    X_avg_matrix = np.zeros((GB_num, t, n_features), dtype=np.float32)  # 存储每个粒球的中心 x_avg
    labels = np.zeros((GB_num, t), dtype=int)  # 存储粒球标签

    # 填充数据
    for GB_idx in range(GB_num):
        for t_step in range(t):
            # 原始数据格式: matrix_list_GB[GB_idx][t_step] = [ (n_features,3)数组, 标签, (n_features,)中心 ]
            matrix_array[GB_idx, t_step] = matrix_list_GB[GB_idx][t_step][0]  # (n_features,3)
            labels[GB_idx, t_step] = matrix_list_GB[GB_idx][t_step][1]  # 标量标签
            X_avg_matrix[GB_idx, t_step] = matrix_list_GB[GB_idx][t_step][2]  # (n_features,)

    n_samples = test_samples.shape[0]
    batch_size = 300  # 根据内存调整
    results = []

    # 分块处理测试样本
    for i in range(0, n_samples, batch_size):
        B_batch = test_samples[i:i + batch_size]
        batch_scores = optimized_process(B_batch, X_avg_matrix, matrix_array, t)
        results.append(batch_scores)

    # 合并结果并找到最近邻
    results = np.vstack(results)  # (n_samples, GB_num)
    results = np.array(results, dtype=np.float32)
    return np.array(results)


def classifier(average_results, matrix_list_GB):
    """
    对动态 IFS 矩阵进行分类（优化版本）
    """
    results_with_t = average_results
    row_indices = []
    k = 1  # 选择 k 个最小值

    # 预提取 matrix_list_GB 中的标签
    labels_all = [matrix_list_GB[idx][0][1] for idx in range(len(matrix_list_GB))]
    labels_all = np.array(labels_all, dtype=int)  # 转换为 numpy 数组

    for arr in results_with_t:
        # 找到 k 个最小值的索引
        min_indices = np.argpartition(arr, k, axis=None)[:k]

        # 直接从预提取的 labels_all 获取标签
        labels = labels_all[min_indices]

        # 统计投票
        most_common_label = Counter(labels).most_common(1)[0][0]
        # most_common_label = np.bincount(labels).argmax()
        row_indices.append(most_common_label)

    return row_indices


def main(random_state, data_x, data_y, labels_len, importances=None):
    feature_seletion = 0

    features_nums = np.zeros(120)
    t = 1
    a = 0
    data_train1, data_test, labels_train, labels_test = preprocess_data(data_x, data_y, random_state)

    for j in range(10, 11):
        features_nums[a] = j

        if feature_seletion == 0:

            model_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            model_rf.fit(data_train1, labels_train)

            # 获取特征重要性分数
            importances = model_rf.feature_importances_

            # 选择k个重要性最高的特征索引
            k = j  # 设定所需特征个数
            beginfeature = k
            sorted_indices = np.argsort(importances)[-k:]  # 获取前k个特征的索引
            data_x = np.array(data_x)
            # 筛选前k个特征
            data_x1 = data_train1[:, sorted_indices]
            data_test1 = data_test[:, sorted_indices]
            print("使用的特征索引", sorted_indices)
            feature_seletion = 1
        else:
            # 选择k个重要性最高的特征索引
            k = j  # 设定所需特征个数
            print("使用的特征数量:", k)
            sorted_indices = np.argsort(importances)[-k:]  # 获取前k个特征的索引
            data_x = np.array(data_x)
            # 筛选前k个特征
            data_x1 = data_train1[:, sorted_indices]
            data_test1 = data_test[:, sorted_indices]
            print("使用的特征索引", sorted_indices)

        n_features = data_x1.shape[1]
        # 合并 data_train和labels_train
        # data_train = scalerdata(data_train)
        # data_test = scalerdata(data_test)
        data_train = np.concatenate((data_x1, labels_train.reshape(-1, 1)), axis=1)
        # 合并 data_train和labels_train
        training_timelen = len(data_train)
        # 输入到粒球
        start_time = time.time()
        Generation_method = 2
        if Generation_method == 2:
            nGB = 20
            n_clusters = []
            # 创建一个空列表
            for _ in range(labels_len):  # 使用下划线(_)表示这个循环变量不会被使用
                n_clusters.append(nGB)  # 每次循环向列表添加数字100
            n_clusters[0] = 200
            n_clusters[1] = 200
            data_GB = CGB_data(data_train, n_clusters)
            print("每个类别的粒球数量", n_clusters)


        matrix_list_GB = []  # 粒球数*t*
        for data_gb in data_GB:  # n个列表里分别有，中心平均值 半径 标签 数据
            data_mean = data_gb[0]  # 数据中心
            data_R = data_gb[1]  # 半径
            data_l = data_gb[2]  # 标签
            data_origin = data_gb[3]  # 数据

            train_lens, n_features = data_origin.shape[0], data_origin.shape[1]
            # 动态 按时间切
            data_train = Time_cut(data_origin, t)
            # 使用 NumPy 数组替代列表
            labels_train1 = np.full(2, data_l, dtype=type(data_l))  # 指定数据类型为 data_l 的类型
            labels_train1 = Time_cut(labels_train1, t)

            # 创建动态直觉模糊矩阵
            matrix_list_all = []

            for i in range(t):
                matrix_list = create_dynamic_if_matrix(
                    data_train[i], labels_train1[i], n_features, data_mean)

                matrix_list_all.append(matrix_list)

            matrix_list_GB.append(matrix_list_all)

        end_time = time.time()
        training_time = end_time - start_time
        print("训练时间：", training_time)
        print("训练时间/训练集：", training_time / training_timelen)

        # 测试集
        start_time = time.time()
        average_results = create_dynamic_if_matrix_test(data_test1, n_features, matrix_list_GB, t)
        end_time = time.time()
        test_matrix_time = end_time - start_time
        print("构建测试集时间：", test_matrix_time)
        print("构建测试集时间/预测集：", test_matrix_time / len(labels_test))

        # 预测
        start_time = time.time()
        predictions_indices = classifier(average_results, matrix_list_GB)  # 换成粒球输入
        end_time = time.time()
        pre_time = end_time - start_time
        print("预测时间：", pre_time)
        print("预测时间/预测集：", pre_time / len(predictions_indices))

        # 评估性能
        accuracy, macro_precision, macro_recall, macro_f1_score, results = evaluate_performance(
            np.array(predictions_indices),
            labels_test)

        Accuracy, Precision, Recall, F1_score, C1_Recall, C1_F1_score, FPR = print_meetrics(a, accuracy, macro_precision, macro_recall, macro_f1_score, results)
        print('特征选择量为：', n_features)
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
    data_x, data_y, labels_len = datasetload()
    for random_state in random_states:
        Accuracy, macro_precision, macro_recall, macro_f1_score, C1_Recall, C1_F1_score, FPR, beginfeature, \
            features_nums = main(random_state, data_x, data_y, labels_len)

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
        print('最大准确率为：', max_accuracy)
        print("最大准确率对应的特征数量为：", max_index)
        non_zero_accuracy = Accuracy[Accuracy != 0]
        print('准确率列表为：', non_zero_accuracy)
        # print("data_GB的数量", len(data_GB))
        print("\n")


    def datemean(resultstemp3, max_index):

        resultstemp = np.mean(resultstemp3, axis=0)
        # Accuracy里的最大值
        max_resultstemp = resultstemp[max_index]

        non_zero_resultstemp = resultstemp[resultstemp != 0]
        return resultstemp, max_resultstemp, non_zero_resultstemp


    print("3次平均结果")
    Accuracy = np.mean(Accuracy3, axis=0) # 多个随机数取平均（所有特征下寻找最大平均值）
    max_accuracy = max(Accuracy)
    max_index = np.argmax(Accuracy)
    max_indextemp = beginfeature + max_index
    non_zero_accuracy = Accuracy[Accuracy != 0]
    print('最大平均准确率为：', max_accuracy)
    print("最大平均准确率对应的特征数量为：", max_indextemp)
    print('平均准确率列表为：', non_zero_accuracy)

    macro_precision, max_macro_precision, non_zero_macro_precision = datemean(macro_precision3, max_index)
    macro_recall, max_macro_recall, non_zero_macro_recall = datemean(macro_recall3, max_index)
    macro_f1_score, max_f1_score, non_zero_f1_score = datemean(macro_f1_score3, max_index)
    C1_Recall, max_C1_Recall, non_zero_C1_Recall = datemean(C1_Recall3, max_index)
    C1_F1_score, max_C1_F1_score, non_zero_C1_F1_score = datemean(C1_F1_score3, max_index)
    FPR, max_FPR, non_zero_FPR = datemean(FPR3, max_index)
    print('最大平均准确率对应的macro_precision为：', max_macro_precision)
    print('最大平均准确率对应macro_recall为：', max_macro_recall)
    print('最大平均准确率对应macro_f1_score为：', max_f1_score)
    print('最大平均准确率对应C1_Recall为：', max_C1_Recall)
    print('最大平均准确率对应C1_F1_score为：', max_C1_F1_score)
    print('最大平均准确率对应FPR为：', max_FPR)
    print('macro_precision列表为：', non_zero_macro_precision)
    print('macro_recall列表为：', non_zero_macro_recall)
    print('macro_f1_score列表为：', non_zero_f1_score)
    print('C1_Recall列表为：', non_zero_C1_Recall)
    print('C1_F1_score列表为：', non_zero_C1_F1_score)
    print('FPR列表为：', non_zero_FPR)
