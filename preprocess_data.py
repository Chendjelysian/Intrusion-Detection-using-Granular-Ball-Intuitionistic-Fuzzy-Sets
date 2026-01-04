import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def datasetload():
    filename = r"G:\数据集\X-IIoTID\X-IIoTID dataset.csv"
    print(filename)
    # 读取数据
    X_data = pd.read_csv(filename, header=None, skiprows=1, nrows=840000)
    # X_data = pd.read_csv(filename, header=None, skiprows=1)
    #  skiprows=[0],, nrows=200000

    # 将最后一列转化为数字类型
    # 攻击类型列表
    attack_names = [
        "Normal", "C&C", "crypto-ransomware", "Exfiltration", "Exploitation", "Lateral _movement",
        "RDOS", "Reconnaissance", "Tampering", "Weaponization"]
    # 创建映射字典
    attack_map = {name: idx for idx, name in enumerate(attack_names)}
    # 将最后一列转化为数字类型
    X_data.iloc[:, -2] = X_data.iloc[:, -2].map(attack_map)

    # 写个循环处理吧，只要数出问题的列就行了
    no_nums_index = [0, 2, 4, 6, 7]

    # 处理非数字的问题列
    for index1 in no_nums_index:
        columns_to_be_mapped = X_data.iloc[:, index1].unique().tolist()
        # 创建映射字典
        attack_map = {name: idx for idx, name in enumerate(columns_to_be_mapped)}
        # 将最后一列转化为数字类型
        X_data.iloc[:, index1] = X_data.iloc[:, index1].map(attack_map)

    # 将所有列尝试转换为数值类型，非数值类型的单元格会变为 NaN
    X_data = X_data.apply(pd.to_numeric, errors='coerce')

    # 选择填充方法，例如使用均值填充
    # X_data.fillna(X_data.mean(), inplace=True)
    X_data.fillna(0, inplace=True)
    print("X_data.shape:", X_data.shape)

    X_data = pd.DataFrame(X_data)
    data_x = X_data.iloc[0:, :-3]  # 调试
    data_y = X_data.iloc[0:, -2]
    print("data_x.shape:", data_x.shape)
    print("data_y.shape:", data_y.shape)
    del X_data
    import gc

    gc.collect()

    # print("现在是多类别")
    data_y = (data_y != 0).astype(int)  # 使用 pandas 向量化操作
    labels_len = len(pd.unique(data_y))

    print("data_y中每个类别的样本数量:", data_y.value_counts())
    return data_x, data_y, labels_len



def preprocess_data(data, labels, random_state, test_size=0.2):
    scaler = MinMaxScaler()
    data = np.array(data)
    data = scaler.fit_transform(data)
    print("数据归一化")

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size,
                                                                        random_state=random_state)

    if not isinstance(data_train, np.ndarray):
        data_train = np.array(data_train)
    if not isinstance(data_test, np.ndarray):
        data_test = np.array(data_test)
    if not isinstance(labels_train, np.ndarray):
        labels_train = np.array(labels_train)
    if not isinstance(labels_test, np.ndarray):
        labels_test = np.array(labels_test)
    del data
    del labels
    return data_train, data_test, labels_train, labels_test
