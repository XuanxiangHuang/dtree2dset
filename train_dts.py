#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Train Classifiers.
#
#
################################################################################
import sys
from decision_tree import DecisionTree
################################################################################


if __name__ == '__main__':
    args = sys.argv[1:]
    train_threshold = 0.7
    test_threshold = 0.7
    if len(args) >= 1 and args[0] == '-bench':
        bench_name = args[1]
        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()
        for ds in name_list:
            ds = ds.strip()
            print(f"################## {ds} ##################")
            dt = DecisionTree()
            acc_train, acc_test = dt.train(f"datasets/{ds}/train.csv", f"datasets/{ds}/test.csv", max_depth=9)

            if acc_train < train_threshold or acc_test < test_threshold:
                print(f'DT: train accuracy {acc_train} < {train_threshold} '
                      f'or test accuracy {acc_test} < {test_threshold}')
            else:
                print(f"DT, Train accuracy: {acc_train * 100.0}%")
                print(f"DT, Test accuracy: {acc_test * 100.0}%")
                print(f"DT, Depth: {dt.max_depth()}")
                print(f"DT, Number of nodes: {dt.num_nodes()}")
                dt.save_model(dt, f"dt_models/{ds}.pkl")
