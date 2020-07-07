from collections import Counter
import pandas as pd
import numpy as np
import csv
import os

iris_path = "./dataset/iris"
labor_path = "./dataset/labor/C4.5"

iris_data = "iris.data"
labor_data = "labor-neg.data"

iris_name_list = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]


def normalize1_min_max(raw_data_list, new_value_range: list):
    if not isinstance(raw_data_list, list):
        raw_data_list = list(raw_data_list)
    if isinstance(raw_data_list[0], str):
        try:
            raw_data_list = [int(v) for v in list(raw_data_list)]
        except ValueError as e:
            return raw_data_list
    max_value = max(raw_data_list)
    min_value = min(raw_data_list)
    new_max = new_value_range[1]
    new_min = new_value_range[0]

    new_list = [(v - min_value) * (new_max - new_min) / (max_value - min_value) + new_min for v in raw_data_list]
    return new_list


def average_mode_fill(raw_data_list):
    """
        This function will use average to fill numerical data,
        and use mode to fill categorical data

        Note: please make sure the missing value is replaced by '?'
    :param raw_data_list:
    :return:
    """
    if not isinstance(raw_data_list, list):
        raw_data_list = list(raw_data_list)

    data_type = "num"
    data_dict = {}
    available_value_list = []
    try:
        available_value_list = [int(v) for v in raw_data_list if v != "?"]
    except ValueError as e:
        data_type = "cat"
        for item in raw_data_list:
            if item == "?":
                continue
            elif item not in data_dict:
                data_dict[item] = 1
            else:
                data_dict[item] += 1

    if data_type == "num":
        average = sum(available_value_list) / len(available_value_list)
        return [v if v != "?" else average for v in raw_data_list]
    elif data_type == "cat":
        mode_category = max(data_dict, key=data_dict.get)
        return [v if v != "?" else mode_category for v in raw_data_list]
    else:
        raise ValueError("Some errors occupied with data_type")


def feature_select(features, label, divide=10):
    feature_num = len(features[0, :])
    label_count = Counter(label)
    label_num = len(label)

    base_entropy = 0
    for i in label_count.keys():
        label_count[i] /= float(label_num)
        base_entropy -= label_count[i] * np.log2(label_count[i])
    IG_dict = {}
    for f in range(feature_num):
        target_feature = features[:, f]
        min_value = min(target_feature)
        max_value = max(target_feature)
        width = (max_value - min_value) / 20

        discrete_feature = np.floor((target_feature - min_value) / width)
        condi_count = Counter(discrete_feature)

        feature_entropy = 0
        for key, value in condi_count.items():
            sub_feature_prob = value / label_num
            sub_label = label[discrete_feature == key]
            sub_label_count = Counter(sub_label)

            sub_entropy = 0
            for k, v in sub_label_count.items():
                condi_prob = v / float(value)
                sub_entropy -= condi_prob * np.log2(condi_prob)
            feature_entropy += sub_feature_prob * sub_entropy

        IG_dict.update({iris_name_list[f]: base_entropy - feature_entropy})

    return IG_dict


def main(mode, dataset):
    if dataset == "iris":
        data_path = os.path.join(iris_path, iris_data)
        save_path = os.path.join(iris_path, "processed_iris.data")
        csv_data = pd.read_csv(data_path, names=iris_name_list)
    elif dataset == "labor":
        data_path = os.path.join(labor_path, labor_data)
        save_path = os.path.join(labor_path, "processed_labor.data")
        csv_data = pd.read_csv(data_path, header=None)
    else:
        raise ValueError("Please make sure dataset in [iris, labor]")

    if mode == "norm":
        csv_data = csv_data.apply(normalize1_min_max, axis=0, args=[[0, 1]])
    elif mode == "fill":
        csv_data = csv_data.apply(average_mode_fill, axis=0)
    elif mode == "feature_select":
        row, col = csv_data.shape
        label = csv_data["class"]
        features = csv_data.drop(["class"], axis=1)
        IG_dict = feature_select(features.values, label, divide=10)
        print(IG_dict)
    else:
        raise ValueError("mode must be one of [norm, fill, feature_select]")

    csv_data.to_csv(save_path, index=False, header=False)


if __name__ == "__main__":
    main("feature_select", "iris")
