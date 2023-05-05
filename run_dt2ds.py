#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Run DT to DS.
#
#
################################################################################
import sys
import time
from Orange.data import Table
from decision_tree import DecisionTree
from tree2rules import Tree2Rules
################################################################################


if __name__ == '__main__':

    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == '-bench':
        bench_name = args[1]
        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()
        for data in name_list:
            data = data.strip()
            if data in ('calendarDOW', 'parity5+5', 'vehicle'):
                # test accuracy too low, < 70%
                continue

            print(f"################## {data} ##################")
            datatable = Table(f"datasets/{data}/train.csv")
            n_features = len([att.name for att in datatable.domain.attributes])
            n_classes = len(datatable.domain.class_var.values)

            dt = DecisionTree.from_file(f"dt_models/{data}.pkl")
            n_nd = dt.num_nodes()
            if n_nd < 10:
                continue

            time_start = time.perf_counter()

            t2r = Tree2Rules(dt, 0)
            paths, axps, ds, path_len_axp_len = t2r.convert_dt_to_ds()
            t2r.save_ds(ds, f"ds_files/{data}.txt")

            time_end = time.perf_counter()
            used_time = time_end - time_start

            len_diff_ratio = [(x - y)/x for (x, y) in path_len_axp_len]

            lit_dt = []
            lit_ds = []
            # collect literals in all paths
            for p in paths:
                lit_dt.append(len(p)-1)
            # collect literals in all rules
            for r in ds:
                lit_ds.append(len(r)-1)

            exp_results = f"{data} & {n_features} & {n_classes} & {dt.max_depth()} & {dt.num_nodes()} & {round(dt.test_acc*100, 1)} & "
            exp_results += f"{len(paths)} & {sum(lit_dt)} & "
            exp_results += f"{len(ds)} & {sum(lit_ds)} & "
            exp_results += f"{round(100*max(len_diff_ratio), 1)} & {round(100*sum(len_diff_ratio)/len(len_diff_ratio), 1)} & "
            exp_results += "{0:.2f}\n".format(used_time)

            with open('dt2ds_results.txt', 'a') as f:
                f.write(exp_results)
