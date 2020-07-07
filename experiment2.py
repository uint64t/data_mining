import pandas as pd
import fp_growth2
from itertools import combinations
from collections import Counter
from operator import itemgetter
import time

sample_data = "./dataset/classification/sampledata_numbers.csv"


def comb(data, length):
    if isinstance(data, set):
        data = list(data)
    data_list = []
    for i in range(len(data)):
        for j in range(i):
            if not isinstance(data[i], int):
                candidate = set(data[i]) | set(data[j])
                if len(candidate) != length or candidate in data_list:
                    continue
                else:
                    data_list.append(candidate)
            else:
                candidate = {data[i], data[j]}
                if candidate not in data_list:
                    data_list.append(candidate)
    data_set = {tuple(v) for v in data_list}
    return data_set


# The unoptimized version of this algorithm is from:
# https://towardsdatascience.com/apriori-algorithm-implementation-using-optimized-approach-with-pandas-a72aacd990fe
# Ths optimized is done by me, by replacing the combination fucntion
def Apriori(dataset_path, support_count=500, optimize=True):
    data = pd.read_csv(dataset_path)

    single_items = (data['items'].str.split(",", expand=True)) \
        .apply(pd.value_counts).sum(axis=1) \
        .where(lambda value: value > support_count).dropna()

    apriori_data = pd.DataFrame(
        {'items': single_items.index.astype(int),
         'support_count': single_items.values,
         'set_size': 1})

    data['set_size'] = data['items'].str.count(",") + 1
    data['items'] = data['items'] \
        .apply(lambda row: set(map(int, row.split(","))))

    items_set = set(single_items.index.astype(int))
    single_items_set = set(single_items.index.astype(int))
    for length in range(2, len(items_set) + 1):
        data = data[data['set_size'] >= length]
        items_set = comb(items_set, length)
        if optimize:
            d = data['items'] \
                .apply(lambda st: pd.Series(s if set(s).issubset(st) else None
                                            for s in items_set))\
                .apply(lambda col: [col.dropna().unique()[0], col.count()] if col.count() >= support_count else None).dropna()
        else:
            d = data['items'] \
                .apply(lambda st: pd.Series(s if set(s).issubset(st) else None
                                            for s in combinations(single_items_set, length))) \
                .apply(lambda col: [col.dropna().unique()[0], col.count()] if
            col.count() >= support_count else None).dropna()

        items_set = {item[0] for item in d.values}
        if d.empty:
            # print("break")
            break

        apriori_data = apriori_data.append(pd.DataFrame(
            {'items': list(map(itemgetter(0), d.values)),
             'support_count': list(map(itemgetter(1), d.values)),
             'set_size': length}), ignore_index=True)

    return apriori_data


def Compare_Apriori(print_result=False):
    support_count = 500
    start = time.time()
    optimized_result = Apriori(sample_data, support_count, True)
    if print_result:
        print(optimized_result)
    print(f"Optimized time: {time.time() - start}")
    start = time.time()
    result = Apriori(sample_data, support_count, False)
    if print_result:
        print(result)
    print(f"Original time: {time.time() - start}")

    validate = []
    for i in optimized_result["items"].values:
        if isinstance(i, int):
            item = [i]
        else:
            item = i
        if set(item) in validate:
            print(f"Dumplicate {item}")
        else:
            validate.append(set(item))
    for i in result["items"].values:
        if isinstance(i, int):
            item = [i]
        else:
            item = i
        if set(item) not in validate:
            print(f"Error: {item}")


def Compare_Apriori_FP(print_result=False):
    support_count = 200
    start = time.time()
    apriori_result = Apriori(sample_data, support_count, True).sort_values(by="support_count", ascending=False)
    if print_result:
        print(apriori_result)
    print(f"Optimized time: {time.time() - start}")
    start = time.time()
    fp_result = fp_growth2.test(sample_data, support_count)
    if print_result:
        print(fp_result)
    print(f"fp time: {time.time() - start}")


def Generate_rule(data, ratio=0.6):
    rule = {}
    for item in data[data["set_size"] > 1]["items"].values:
        condi_count = data[data["items"] == item]["support_count"].values[0]
        for length in range(1, len(item)):
            for array in combinations(item, length):
                if len(array) == 1:
                    array = array[0]
                whole_count = data[data["items"] == array]["support_count"].values
                if len(whole_count) == 0:
                    whole_count = data[data["items"] == (array[1], array[0])]["support_count"].values

                if condi_count > whole_count[0] * ratio:
                    rule[f"{array}->{item}"] = condi_count / whole_count[0]
    return rule


def main():
    fp_result = fp_growth2.test(sample_data, 200)
    rule = Generate_rule(fp_result)
    print(rule)


if __name__ == '__main__':
    main()
    # Compare_Apriori_FP(print_result=True)
    # Compare_Apriori()
