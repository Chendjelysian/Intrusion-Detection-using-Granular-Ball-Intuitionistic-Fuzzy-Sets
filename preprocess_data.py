import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def datasetload(filename):
    # allowed_dir = os.path.abspath(os.path.join(os.getcwd(), "data"))
    # safe_filename = os.path.abspath(filename)
    #
    # try:
    #     common_path = os.path.commonpath([allowed_dir, safe_filename])
    #     if common_path != allowed_dir:
    #         raise ValueError("The file path is not within the allowed directory range.")
    # except ValueError:
    #     raise ValueError("The file path is not within the allowed directory range.")
    safe_filename = filename

    if not os.path.exists(safe_filename):
        raise FileNotFoundError(f"The file {safe_filename} does not exist.")

    print("The file being read is", filename)
    if safe_filename == r"data\X-IIOTID\X-IIoTID dataset.csv":

        X_data = pd.read_csv(safe_filename, header=None, skiprows=1, nrows=840000)

        attack_names = [
            "Normal", "C&C", "crypto-ransomware", "Exfiltration", "Exploitation", "Lateral _movement",
            "RDOS", "Reconnaissance", "Tampering", "Weaponization"]
        # map
        attack_map = {name: idx for idx, name in enumerate(attack_names)}

        X_data.iloc[:, -2] = X_data.iloc[:, -2].map(attack_map)

        no_nums_index = [0, 2, 4, 6, 7]

        for index1 in no_nums_index:
            columns_to_be_mapped = X_data.iloc[:, index1].unique().tolist()

            attack_map = {name: idx for idx, name in enumerate(columns_to_be_mapped)}

            X_data.iloc[:, index1] = X_data.iloc[:, index1].map(attack_map)

        X_data = X_data.apply(pd.to_numeric, errors='coerce')

        X_data.fillna(0, inplace=True)
        print("X_data.shape:", X_data.shape)

        X_data = pd.DataFrame(X_data)
        data_x = X_data.iloc[0:, :-3]
        data_y = X_data.iloc[0:, -2]
        # print("data_x.shape:", data_x.shape)
        # print("data_y.shape:", data_y.shape)
        del X_data
        import gc
        gc.collect()
        data_y = (data_y != 0).astype(int)
        labels_len = len(pd.unique(data_y))
        print("Number of samples for each category in data_y:", data_y.value_counts())
        return data_x, data_y, labels_len

    elif safe_filename == r"data\TON-IOT\Train_Test_Network.csv":

        X_data = pd.read_csv(filename, skiprows=1, header=None, nrows=500 * 1000)
        no_nums_index = [1, 3, 5, 6, 10, 20, 21, 22, 40, 44, 16, 23, 24, 25, 26, 27, 28, 29,
                         30, 31, 32, 33, 37, 38, 39, 41, 42]

        for index1 in no_nums_index:
            columns_to_be_mapped = X_data.iloc[:, index1].unique().tolist()
            attack_map = {name: idx for idx, name in enumerate(columns_to_be_mapped)}
            X_data.iloc[:, index1] = X_data.iloc[:, index1].map(attack_map)

        X_data = X_data.apply(pd.to_numeric, errors='coerce')

        X_data.fillna(0, inplace=True)

        print("X_data.shape:", X_data.shape)
        X_data = pd.DataFrame(X_data)

        data_x = X_data.iloc[:, :-1]
        data_y = X_data.iloc[:, -1]
        print("data_x.shape:", data_x.shape)
        print("data_y.shape:", data_y.shape)

        data_y = (data_y != 0).astype(int)
        labels_len = len(pd.unique(data_y))
        print("Number of samples for each category in data_y:", data_y.value_counts())
        return data_x, data_y, labels_len
    elif safe_filename == r"data\NSLKDD\KDDTest+.csv":
        Y_data = pd.read_csv(filename, header=None)

        def preprocess_data_only(X_data):
            no_nums_index = [1, 2, 3, -2]

            for index1 in no_nums_index:
                columns_to_be_mapped = X_data.iloc[:, index1].unique().tolist()
                attack_map = {name: idx for idx, name in enumerate(columns_to_be_mapped)}
                X_data.iloc[:, index1] = X_data.iloc[:, index1].map(attack_map)
            X_data = X_data.apply(pd.to_numeric, errors='coerce')

            X_data.fillna(0, inplace=True)

            print("X_data.shape:", X_data.shape)
            return X_data

        Y_data = preprocess_data_only(Y_data)

        data_test = Y_data.iloc[0:, :-2]
        labels_test = Y_data.iloc[0:, -2]
        data_lens = data_test.shape[0]
        data_x = data_test
        data_y = labels_test

        print("data_x.shape:", data_x.shape)
        print("data_y.shape:", data_y.shape)
        del Y_data
        import gc
        gc.collect()

        data_y = (data_y != 0).astype(int)
        labels_len = len(pd.unique(data_y))
        print("Number of samples for each category in data_y:", data_y.value_counts())
        return data_x, data_y, labels_len
    elif safe_filename == r"data\WUSTL-IIOT\wustl_iiot_2021.csv":

        X_data = pd.read_csv(filename, header=None, skiprows=1, nrows=1200000)

        no_nums_index = [2, 3, -2]
        for index1 in no_nums_index:
            columns_to_be_mapped = X_data.iloc[:, index1].unique().tolist()
            attack_map = {name: idx for idx, name in enumerate(columns_to_be_mapped)}
            X_data.iloc[:, index1] = X_data.iloc[:, index1].map(attack_map)
        X_data = X_data.iloc[:, 2:-1]
        X_data = X_data.apply(pd.to_numeric, errors='coerce')
        X_data.fillna(0, inplace=True)
        print("X_data.shape:", X_data.shape)
        X_data = pd.DataFrame(X_data)

        data_x = X_data.iloc[:, :-1]
        data_y = X_data.iloc[:, -1]
        print("data_x.shape:", data_x.shape)
        print("data_y.shape:", data_y.shape)
        del X_data
        import gc
        gc.collect()

        data_y = (data_y != 0).astype(int)
        labels_len = len(pd.unique(data_y))
        print("Number of samples for each category in data_y:", data_y.value_counts())
        return data_x, data_y, labels_len
    elif safe_filename == r"data\KDDCUP99\kddcup.data":

        X_data = pd.read_csv(filename, header=None, nrows=5000000)
        no_nums_index = [1, 2, 3, -1]

        for index1 in no_nums_index:
            columns_to_be_mapped = X_data.iloc[:, index1].unique().tolist()
            attack_map = {name: idx for idx, name in enumerate(columns_to_be_mapped)}
            X_data.iloc[:, index1] = X_data.iloc[:, index1].map(attack_map)

        for col in X_data.columns:
            if X_data[col].map(type).nunique() > 1:
                print(f'Column {col} contains mixed types')
        X_data = X_data.apply(pd.to_numeric, errors='coerce')

        nan_counts = X_data.isna().sum()
        print(nan_counts[nan_counts > 0])

        X_data.fillna(0, inplace=True)
        nan_counts_after_fill = X_data.isna().sum()
        print(nan_counts_after_fill[nan_counts_after_fill > 0])
        print("X_data1.shape:", X_data.shape)

        X_data = pd.DataFrame(X_data)

        data_x = X_data.iloc[0:, :-1]
        data_y = X_data.iloc[0:, -1]
        print("data_x.shape:", data_x.shape)
        print("data_y.shape:", data_y.shape)

        del X_data
        import gc
        gc.collect()

        data_y = (data_y != 0).astype(int)
        labels_len = len(pd.unique(data_y))
        print("Number of samples for each category in data_y:", data_y.value_counts())
        return data_x, data_y, labels_len
    elif safe_filename == r"data\UNSW-NB15\UNSW-NB15_full.csv":

        X_data = pd.read_csv(filename, header=None, nrows=2700001)

        def preprocess_data_only(X_data):

            no_nums_index = [0, 1, 2, 3, 4, 5, 13, 24, 37, 38, 39, 47, -2]

            for index1 in no_nums_index:
                columns_to_be_mapped = X_data.iloc[:, index1].unique().tolist()

                attack_map = {name: idx for idx, name in enumerate(columns_to_be_mapped)}

                X_data.iloc[:, index1] = X_data.iloc[:, index1].map(attack_map)

            X_data = X_data.apply(pd.to_numeric, errors='coerce')

            X_data.fillna(0, inplace=True)
            X_data.dropna(inplace=True)
            print("X_data.shape:", X_data.shape)

            return X_data

        X_data = preprocess_data_only(X_data)

        data_x = X_data.iloc[0:, :-1]
        data_y = X_data.iloc[0:, -1]

        print("data_x.shape:", data_x.shape)
        print("data_y.shape:", data_y.shape)
        del X_data
        import gc
        gc.collect()
        data_y = (data_y != 0).astype(int)

        labels_len = len(pd.unique(data_y))
        print("Number of samples for each category in data_y:", data_y.value_counts())
        return data_x, data_y, labels_len



def preprocess_data(data, labels, random_state, test_size=0.2):
    scaler = MinMaxScaler()
    data = np.array(data)
    data = scaler.fit_transform(data)
    print("Data normalization")

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
