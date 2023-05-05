#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Splitting datasets, save .csv to be compatible with Orange3
#
#
################################################################################
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
################################################################################


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == '-bench':
        bench_name = args[1]
        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()
        for ds in name_list:
            ds = ds.strip()
            print(f"################## {ds} ##################")
            input_file = f'datasets/{ds}/{ds}.csv'
            df = pd.read_csv(input_file, sep=',')
            train_df, test_df = train_test_split(df, test_size=0.2)

            shape = train_df.shape
            train_df.loc[-1] = [''] * (shape[1] - 1) + ['class']
            train_df.index = train_df.index + 1
            train_df.sort_index(inplace=True)
            train_df.loc[-1] = ['continuous'] * (shape[1] - 1) + ['discrete']
            train_df.index = train_df.index + 1
            train_df.sort_index(inplace=True)

            shape = test_df.shape
            test_df.loc[-1] = [''] * (shape[1] - 1) + ['class']
            test_df.index = test_df.index + 1
            test_df.sort_index(inplace=True)
            # we assume all the features are continuous,
            # but this is not true, so we check the datasets with PMLB description file.
            # However, Orange3 reports that
            # "ValueError: Exhaustive binarization does not handle attributes with more than 16 values"
            # which means we cannot mark features having too many values as discrete.
            # and it implies that in this case, Orange3 will prefer to use multi-way split,
            # but then it is no longer a binary decision tree.
            # so, for age, we should mark it as continuous.
            test_df.loc[-1] = ['continuous'] * (shape[1] - 1) + ['discrete']
            test_df.index = test_df.index + 1
            test_df.sort_index(inplace=True)

            train_df.to_csv(f'datasets/{ds}/train.csv', sep=',', index=False)
            test_df.to_csv(f'datasets/{ds}/test.csv', sep=',', index=False)
