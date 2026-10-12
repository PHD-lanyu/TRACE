import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def process_results_folder(root_folder):
    methods = [name for name in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, name)) and not name.startswith(".")]
    dataset_results = defaultdict(lambda: defaultdict(list))
    dataset_is_cf = defaultdict(list)
    for method in methods:
        method_path = os.path.join(root_folder, method)
        for filename in os.listdir(method_path):
            if not filename.endswith(".json"):
                continue
            dataset_name = os.path.splitext(filename)[0]
            file_path = os.path.join(method_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            correct_values, is_cf_values = [], []
            for result in data.get("results", []):
                correct_values.append(bool(result.get("correct", False)))
                if "is_cf" in result:
                    is_cf_values.append(bool(result.get("is_cf", False)))
            dataset_results[dataset_name][method] = correct_values
            if not dataset_is_cf[dataset_name] and is_cf_values:
                dataset_is_cf[dataset_name] = is_cf_values
    return dataset_results, dataset_is_cf, methods


def create_summary_tables(dataset_results, dataset_is_cf):
    tables = {}
    for dataset_name, methods_data in dataset_results.items():
        max_samples = max(len(values) for values in methods_data.values())
        rows = {method: values + [None] * (max_samples - len(values)) for method, values in methods_data.items()}
        if dataset_is_cf[dataset_name]:
            values = dataset_is_cf[dataset_name]
            rows["is_cf"] = values + [None] * (max_samples - len(values))
        tables[dataset_name] = pd.DataFrame(rows)
    return tables


def save_tables_to_excel(tables, output_file):
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for dataset_name, df in tables.items():
            df.to_excel(writer, sheet_name=dataset_name[:31], index_label="sample_index")


def calculate_metrics_for_dataset(df, dataset_name, methods):
    if "is_cf" not in df.columns:
        if "rag_prompting" in df.columns:
            df["is_cf"] = df["rag_prompting"].apply(lambda value: not bool(value))
        else:
            return None
    if "query_only" not in df.columns:
        return None

    subsets = {
        "SCTI": df[df["query_only"] == True],
        "SCFI": df[df["query_only"] == False],
        "ACC_Te": df[df["is_cf"] == False],
        "ACC_Fe": df[df["is_cf"] == True],
        "ACC_TiTe": df[(df["query_only"] == True) & (df["is_cf"] == False)],
        "ACC_TiFe": df[(df["query_only"] == True) & (df["is_cf"] == True)],
        "ACC_FiFe": df[(df["query_only"] == False) & (df["is_cf"] == True)],
        "ACC_FiTe": df[(df["query_only"] == False) & (df["is_cf"] == False)],
        "ACC_TIuTe": df[(df["query_only"] == True) | (df["is_cf"] == False)],
    }
    rows = []
    for method in [name for name in methods if name in df.columns]:
        row = {"dataset": dataset_name, "column": method, "n_total": len(df)}
        for name, subset in subsets.items():
            row[name] = subset[method].mean() if len(subset) else np.nan
            row["n_" + name.replace("ACC_", "").replace("SC", "")] = len(subset)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    parser.add_argument("--summary-xlsx", default="results_summary.xlsx")
    parser.add_argument("--metrics-csv", default="all_accuracy_metrics.csv")
    args = parser.parse_args()
    dataset_results, dataset_is_cf, methods = process_results_folder(args.root)
    if not dataset_results:
        print("No result json files found.")
        return
    tables = create_summary_tables(dataset_results, dataset_is_cf)
    save_tables_to_excel(tables, args.summary_xlsx)
    metric_tables = [calculate_metrics_for_dataset(df, name, methods) for name, df in tables.items()]
    metric_tables = [table for table in metric_tables if table is not None and not table.empty]
    if metric_tables:
        pd.concat(metric_tables, ignore_index=True).to_csv(args.metrics_csv, index=False)
    print(f"Saved {args.summary_xlsx}" + (f" and {args.metrics_csv}" if metric_tables else ""))


if __name__ == "__main__":
    main()
