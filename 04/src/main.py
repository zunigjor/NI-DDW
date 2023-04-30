import pandas as pd
from collections import Counter


def remove_short_visits(df, max_time):
    print("=========================== MERGED SHAPE ===========================")
    print(f"{df.shape[0]} rows x {df.shape[1]} columns")
    df = df[df["Length_seconds"] > max_time]
    print("================== MERGED WITHOUT SHORT VISITS SHAPE ===============")
    print(f"{df.shape[0]} rows x {df.shape[1]} columns")
    return df


def get_conversions(df):
    # main conversions
    main_conv = df[df["PageName"].isin(["APPLICATION", "CATALOG"])]
    print("========================= MAIN CONVERSIONS =========================")
    print(main_conv)
    # micro conversions
    micro_conv = df[df["PageName"].isin(["DISCOUNT", "HOWTOJOIN", "INSURANCE", "WHOWEARE"])]
    print("========================= MICRO CONVERSIONS ========================")
    print(micro_conv)
    return main_conv, micro_conv


def get_apriori_dataset(main_conversions, clicks, visitors):
    all_main_conv = pd.merge(main_conversions, clicks, how="left", on="VisitID")
    print("===================== ALL USER VISITED PAGES =======================")
    print(all_main_conv)
    dataset = []
    for visitor in visitors["VisitID"]:
        p = all_main_conv[all_main_conv["VisitID"] == visitor]
        if p.size > 0:
            val = p['PageName_y'].values[:]
            dataset.append(val)
    return dataset


def frequentItems(transactions, support):
    counter = Counter()
    for trans in transactions:
        counter.update(frozenset([t]) for t in trans)
    return set(item for item in counter if counter[item] / len(transactions) >= support), counter


def generateCandidates(L, k):
    candidates = set()
    for a in L:
        for b in L:
            union = a | b
            if len(union) == k and a != b:
                candidates.add(union)
    return candidates


def filterCandidates(transactions, itemsets, support):
    counter = Counter()
    for trans in transactions:
        subsets = [itemset for itemset in itemsets if itemset.issubset(trans)]
        counter.update(subsets)
    return set(item for item in counter if counter[item] / len(transactions) >= support), counter


def apriori(transactions, support):
    result = []
    result_count = Counter()
    candidates, counter = frequentItems(transactions, support)
    result += candidates
    result_count += counter
    k = 2
    while candidates:
        candidates = generateCandidates(candidates, k)
        candidates,counter = filterCandidates(transactions, candidates, support)
        result += candidates
        result_count += counter
        k += 1
    result_count = {item:(result_count[item]/len(transactions)) for item in result_count}
    return result, result_count


if __name__ == "__main__":
    clicks = pd.read_csv("./data/clicks.csv")
    search_engine_map = pd.read_csv("./data/search_engine_map.csv")
    visitors = pd.read_csv("./data/visitors.csv")
    print("======================== CLICKS DESCRIPTION ========================")
    print(clicks.describe())
    print("================== SEARCH ENGINE MAP DESCRIPTION  ==================")
    print(search_engine_map.describe())
    print("======================= VISITORS DESCRIPTION  ======================")
    print(visitors.describe())
    # Change time on page to intervals (cleanup)
    clicks["TimeOnPage"] = pd.cut(clicks["TimeOnPage"], 50)
    # Remove LocalID, not needed
    del clicks["LocalID"]
    # Merge tables
    df = pd.merge(
        pd.merge(clicks, visitors, how="left", on="VisitID"),
        search_engine_map, how="left", on="Referrer"
    )
    # Remove visits which are too short
    df = remove_short_visits(df, 30)
    main_conversions, micro_conversions = get_conversions(df)
    main_conversions.to_csv("../results/main_conversions.csv")
    micro_conversions.to_csv("../results/micro_conversions.csv")
    # Extract dataset for apriori with focus on main_conversions
    dataset = get_apriori_dataset(main_conversions, clicks, visitors)
    # Use apriori algorithm
    frequent_item_sets, supports = apriori(dataset, 0.05)
    item_value_pairs = [(fis, supports[fis]) for fis in frequent_item_sets]
    item_value_pairs_sorted = sorted(item_value_pairs, key=lambda x: x[1], reverse=True)
    print("============================== APRIORI =============================")
    for ivps in item_value_pairs_sorted:
        print(f"{ivps[0]} - {ivps[1]}")
    exit(0)
